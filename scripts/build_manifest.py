#!/usr/bin/env python3
"""Build or update the dataset manifest """

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from faceswap.data.alignment import FaceAligner, InsightFaceDetector
from faceswap.data.manifest import ManifestBuilder
from faceswap.data.quality import QualityThresholds 


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Build or update the dataset manifest.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--lfw-root", default=None, help="override data.lfw_root from config")
    ap.add_argument("--update", default=None, metavar="NEW_ROOT",
                    help="add images from NEW_ROOT to the existing manifest instead of rebuilding")
    ap.add_argument("--limit", type=int, default=None, help="process at most N images (smoke test)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["data"]
    q = data_cfg["quality"]

    detector = InsightFaceDetector()
    aligner = FaceAligner(detector=detector, output_size=data_cfg["image_size"])
    thresholds = QualityThresholds(
        min_det_score=q["min_det_score"],
        min_blur=q["min_blur"],
        min_interocular=q["min_interocular"],
        min_frontality=q["min_frontality"],
    )
    builder = ManifestBuilder(aligner, thresholds, aligned_root=data_cfg["aligned_root"])

    manifest_path = data_cfg["manifest"]
    if args.update:
        builder.update(manifest_path, args.update,
                       image_size=data_cfg["image_size"], limit=args.limit)
    else:
        lfw_root = args.lfw_root or data_cfg["lfw_root"]
        qr_path = str(Path(manifest_path).with_name("quality_report.json"))
        builder.build(
            lfw_root, manifest_path,
            source_meta={"name": "LFW deep-funneled", "kaggle_dataset": "atulanandjha/lfwpeople"},
            image_size=data_cfg["image_size"],
            quality_report_path=qr_path,
            limit=args.limit,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

