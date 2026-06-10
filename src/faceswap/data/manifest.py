"""Building the dataset manifest.
This is the core of the image dataset control of what the model trains/evals on. Once checked, json report.
Reasons it's set up this way: reproducibility, the alignment happens once not every epoch.
quality report logs metrics for every scanned face, accepted or not.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional
import cv2

from .alignment import FaceAligner
from .quality import QualityThresholds, assess

log = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


########################
# Data structures
########################
@dataclass
class ManifestEntry:
    #One accepted face

    id: str                      # "<identity>/<stem>", stable across runs
    identity: str
    source_path: str
    aligned_path: str
    source_sha256: str           # hash of original bytes
    aligned_sha256: str          # hash of the aligned crop (reproducibility)
    det_score: float
    blur: float
    interocular: float
    frontality: float
    bbox: Optional[list[float]]
    landmarks: list[list[float]]


@dataclass
class ScanRow:
    #Per-face record for the quality report

    id: str
    identity: str
    det_score: float
    blur: float
    interocular: float
    frontality: float
    accepted: bool
    reasons: list[str] = field(default_factory=list)


########################
# Small IO helpers
########################
def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path:Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, obj: dict) -> None:
    # temp file + rename, so a crash can't leave a half-written manifest behind
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


########################
# Builder
########################
class ManifestBuilder:
    #Build and updates manifests from an LFW-style image tree

    def __init__(self, aligner: FaceAligner, thresholds: QualityThresholds, aligned_root: str | Path):
        self.aligner = aligner
        self.thresholds = thresholds
        self.aligned_root = Path(aligned_root)

    #scanning
    @staticmethod
    def iter_images(lfw_root: str | Path) -> Iterator[tuple[str, Path]]:
        """returns (identity, image_path) for LFW tree
        (<root>/<Name>/<Name>_0001.jpg). sorted so the manifest is stable."""
        root = Path(lfw_root)
        
        for identity_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for img in sorted(identity_dir.iterdir()):
                if img.suffix.lower() in _IMAGE_EXTS:
                    yield identity_dir.name, img

    # per-image work
    def process_image(self, identity: str, path: Path) -> tuple[Optional[ManifestEntry], ScanRow]:
        # Detect, align, quality-gate
        entry_id = f"{identity}/{path.stem}"
        image = cv2.imread(str(path))
        
        if image is None:
            return None, ScanRow(entry_id, identity, 0, 0, 0, 0, False, ["unreadable"])

        aligned = self.aligner.align(image)
        if aligned is None:
            return None, ScanRow(entry_id, identity, 0, 0, 0, 0, False, ["no_face"])

        report = assess(aligned, self.thresholds)
        row = ScanRow(
            id=entry_id,
            identity=identity,
            det_score=report.det_score,
            blur=report.blur,
            interocular=report.interocular,
            frontality=report.frontality,
            accepted=report.accepted,
            reasons=report.reasons,
        )
        if not report.accepted:
            return None, row

        # save PNG
        out_path = self.aligned_root / identity / f"{path.stem}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), aligned.image)

        entry = ManifestEntry(
            id=entry_id,
            identity=identity,
            source_path=str(path),
            aligned_path=str(out_path),
            source_sha256=_sha256_file(path),
            aligned_sha256=_sha256_file(out_path),
            det_score=report.det_score,
            blur=report.blur,
            interocular=report.interocular,
            frontality=report.frontality,
            bbox=None if aligned.bbox is None else [float(x) for x in aligned.bbox],
            landmarks=aligned.landmarks.tolist(),
        )
        return entry, row

    # assembly
    def _assemble(self, entries: list[ManifestEntry],source_meta: dict, revision: int, image_size: int,) -> dict:
        #Accepted entries -> the manifest dict (with the identity index)

        identities: dict[str, list[str]] = {}
        for e in entries:
            identities.setdefault(e.identity, []).append(e.id)

        multi = sum(1 for ids in identities.values() if len(ids) >= 2)
        return {
            "schema_version": SCHEMA_VERSION,
            "revision": revision,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "source": source_meta,
            "config": {
                "image_size": image_size,
                "quality_thresholds": asdict(self.thresholds),
            },
            "stats": {
                "faces_accepted": len(entries),
                "identities_total": len(identities),
                "identities_with_multiple_images": multi,
            },
            # pairing use this index to find same-identity pairs. the
            # single-image problem is just how many of these lists have len >= 2
            "identities": identities,
            "entries": [asdict(e) for e in entries],
        }

    def build(self, lfw_root: str | Path, manifest_path: str | Path, source_meta: Optional[dict] = None,image_size: int = 256, quality_report_path: Optional[str | Path] = None,
                limit: Optional[int] = None,) -> dict:
        
        #Build from scratch over an LFW tree

        entries: list[ManifestEntry] = []
        scan: list[ScanRow] = []
        rejection_breakdown: dict[str, int] = {}

        for i, (identity, path) in enumerate(self.iter_images(lfw_root)):
            if limit is not None and i >= limit:
                break
            entry, row = self.process_image(identity, path)
            scan.append(row)
            if entry is not None:
                entries.append(entry)
            else:
                for reason in row.reasons:
                    rejection_breakdown[reason] = rejection_breakdown.get(reason, 0) + 1
            if (i + 1) % 500 == 0:
                log.info("scanned %d images, %d accepted", i + 1, len(entries))

        source_meta = dict(source_meta or {})
        source_meta.setdefault("root", str(lfw_root))
        manifest = self._assemble(entries, source_meta, revision=1, image_size=image_size)
        manifest["stats"]["images_scanned"] = len(scan)
        manifest["stats"]["rejection_breakdown"] = rejection_breakdown

        _write_json(Path(manifest_path), manifest)
        if quality_report_path is not None:
            _write_json(
                Path(quality_report_path),
                {"schema_version": SCHEMA_VERSION, "scan": [asdict(r) for r in scan]},
            )
        log.info(
            "manifest written: %d/%d faces accepted across %d identities",
            len(entries),
            len(scan),
            manifest["stats"]["identities_total"],
        )
        return manifest

    def update(self, manifest_path: str | Path, new_root: str | Path, image_size: int = 256, limit: Optional[int] = None,) -> dict:
        """
        Add images from new_root into an existing manifest. the task's dataset-update method
        """
        manifest_path = Path(manifest_path)
        with open(manifest_path) as f:
            manifest = json.load(f)

        existing_entries =[ManifestEntry(**e) for e in manifest["entries"]]
        seen_hashes = {e.source_sha256 for e in existing_entries}
        seen_ids = {e.id for e in existing_entries}

        added = 0
        for i, (identity, path) in enumerate(self.iter_images(new_root)):
            if limit is not None and i >= limit:
                break
            # cheap hash check before doing detection work
            src_hash = _sha256_file(path)
            entry_id = f"{identity}/{path.stem}"
            if src_hash in seen_hashes or entry_id in seen_ids:
                continue
            entry, _ = self.process_image(identity, path)
            if entry is not None:
                existing_entries.append(entry)
                seen_hashes.add(entry.source_sha256)
                seen_ids.add(entry.id)
                added += 1

        new_revision = int(manifest.get("revision", 1)) + 1
        source_meta = dict(manifest.get("source", {}))
        source_meta["last_update_root"] = str(new_root)
        updated = self._assemble(
            existing_entries, source_meta, revision=new_revision, image_size=image_size
        )
        # keep the original scan count, record the delta
        updated["stats"]["images_scanned"] = manifest.get("stats", {}).get(
            "images_scanned", len(existing_entries)
        )
        updated["stats"]["added_this_revision"] = added
        _write_json(manifest_path, updated)
        log.info("update complete: +%d faces, revision %d", added, new_revision)
        
        return updated
    

    