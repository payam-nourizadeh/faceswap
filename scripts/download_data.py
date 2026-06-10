#!/usr/bin/env python3
"""Download and verify the LFW dataset.

Usage:
    python scripts/download_data.py --dest data
    python scripts/download_data.py --dest data --no-download   # if already placed

Falls back gracefully if Kaggle isn't available; prints manual download instructions
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faceswap.data.acquire import (  # noqa: E402
    download_via_kaggle,
    extract_archives,
    locate_lfw_root,
    summarize,
)

KAGGLE_DATASET = "atulanandjha/lfwpeople"
KAGGLE_URL = "https://www.kaggle.com/datasets/atulanandjha/lfwpeople"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Download/verify the LFW dataset.")
    ap.add_argument("--dest", default="data", help="directory to download/extract into")
    ap.add_argument("--no-download", action="store_true",
                    help="skip Kaggle download; just extract+locate what is present")
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    root = locate_lfw_root(dest)
    if root is None and not args.no_download:
        logging.info("Attempting Kaggle download of %s ...", KAGGLE_DATASET)
        if not download_via_kaggle(KAGGLE_DATASET, dest):
            logging.info(
                "\nAutomated download unavailable. Please download manually:\n"
                "  1) Get the dataset from: %s\n"
                "  2) Unzip it under: %s\n"
                "  3) Re-run with --no-download\n",
                KAGGLE_URL, dest.resolve(),
            )
    extract_archives(dest)
    root = locate_lfw_root(dest)

    if root is None:
        logging.error("Could not locate an LFW image tree under %s.", dest.resolve())
        return 1

    info = summarize(root)
    logging.info(
        "\nLFW located at: %s\n  identities: %d\n  images: %d\n"
        "  identities with >=2 images: %d  (<- the trainable-pairs population)",
        info["root"], info["identities"], info["images"],
        info["identities_with_multiple_images"],
    )
    logging.info("\nNext: python scripts/build_manifest.py --config configs/default.yaml "
                 "--lfw-root %s", info["root"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

