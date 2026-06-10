"""Helpers for finding and downloading the LFW dataset."""

from __future__ import annotations

import logging
import re
import tarfile
import zipfile
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
# LFW filenames look like Aaron_Eckhart_0001.jpg
_LFW_FILE = re.compile(r".+_\d{4}\.(jpg|jpeg|png|bmp)$", re.IGNORECASE)


def extract_archives(base: str | Path) -> None:
    """Extract any tar/zip archives found"""
    
    base = Path(base)
    for archive in sorted(base.rglob("*")):
        suffixes = "".join(archive.suffixes).lower()
        try:
            if suffixes.endswith((".tar.gz", ".tgz", ".tar")):
                with tarfile.open(archive) as tf:
                    tf.extractall(archive.parent, filter="data")
                log.info("extracted %s", archive.name)
            elif suffixes.endswith(".zip"):
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(archive.parent)
                log.info("extracted %s", archive.name)
        except (tarfile.TarError, zipfile.BadZipFile) as exc:
            log.warning("could not extract %s: %s", archive.name, exc)


def locate_lfw_root(base: str | Path) -> Optional[Path]:
    # kaggle download unzips to different folder names depending on version, so can't hardcode it. Look for the identity pattern

    base = Path(base)
    if not base.exists():
        return None

    candidates: dict[Path, set[str]] = {}
    for path in base.rglob("*"):
        if path.is_file() and _LFW_FILE.match(path.name):
            root = path.parent.parent
            candidates.setdefault(root, set()).add(path.parent.name)

    if not candidates:
        return None
    return max(candidates.items(), key=lambda kv: len(kv[1]))[0]


def summarize(lfw_root: str | Path) -> dict:
    # report summary identities and images under a located LFW root
    
    lfw_root = Path(lfw_root)
    identities = [d for d in lfw_root.iterdir() if d.is_dir()]
    image_count = 0
    multi= 0
    for d in identities:
        imgs = [f for f in d.iterdir() if f.suffix.lower() in _IMAGE_EXTS]
        image_count += len(imgs)
        if len(imgs) >= 2:
            multi += 1
    
    return {
        "root": str(lfw_root),
        "identities": len(identities),
        "images": image_count,
        "identities_with_multiple_images": multi,
    }


def download_via_kaggle(dataset: str, dest: str | Path) -> bool:
    # Needs the kaggle CLI + ~/.kaggle/kaggle.json. Returns False on any failure like corporate firewall, so the caller can fall back to printing manual download instructions.
    import subprocess

    dest= Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset, "-p", str(dest), "--unzip"],
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        log.warning("kaggle download failed (%s)", exc)
        return False
    
