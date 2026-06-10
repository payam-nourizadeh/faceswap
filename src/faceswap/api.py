"""FastAPI endpoint + CLI for the face-swap pipeline.

CLI:  python -m faceswap.api cli --source s.jpg --target t.jpg --out o.jpg
TO DO: HTTP: POST /swap

"""
import argparse
import logging
from typing import Optional
import cv2
import numpy as np

log = logging.getLogger(__name__)

def _decode_image(data: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img  # None if bytes are not a decodable image


def _encode_png(image_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("failed to encode result PNG")
    return buf.tobytes()


def create_app(swapper):

    # TO DOOOOOOOOO
    #Build the fastAPI app around trained swapper.

    from fastapi import FastAPI
    from fastapi.responses import Response

    return None


def run_cli(args: argparse.Namespace) -> int:
    import yaml
    from .inference import load_swapper

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    swapper = load_swapper(cfg, args.checkpoint, blend_method=args.blend)

    source = cv2.imread(args.source)
    target = cv2.imread(args.target)
    if source is None or target is None:
        log.error("could not read source or target image")
        return 1
    result = swapper.swap_image(source, target, all_faces=args.all_faces)
    cv2.imwrite(args.out, result)
    log.info("wrote %s", args.out)
    return 0


def run_video(args: argparse.Namespace) -> int:
    import yaml
    from .inference import load_swapper
    from .video import process_video

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    swapper = load_swapper(cfg, args.checkpoint, blend_method=args.blend)

    source = cv2.imread(args.source)
    if source is None:
        log.error("could not read source image")
        return 1
    vcfg = cfg.get("video", {})
    stats = process_video(
        swapper, source, args.input, args.out,
        working_size=args.working_size or vcfg.get("working_size", 128),
        detect_every=args.detect_every or vcfg.get("detect_every", 5),
        smoothing_alpha=vcfg.get("smoothing_alpha", 0.6),
        blend_method=args.blend,
    )
    log.info("done: %s", stats)
    return 0


def run_serve(args: argparse.Namespace) -> int:

    # TO DOOOOOOOOOO
    import uvicorn
    import yaml

    from .inference import load_swapper

    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Face swap CLI")
    sub = ap.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/default.yaml")
    common.add_argument("--checkpoint", required=True)
    common.add_argument("--blend", default="alpha", choices=["alpha", "poisson"])

    p_cli = sub.add_parser("cli", parents=[common], help="swap a single image pair")
    p_cli.add_argument("--source", required=True)
    p_cli.add_argument("--target", required=True)
    p_cli.add_argument("--out", required=True)
    p_cli.add_argument("--all-faces", action="store_true", help="swap every face in target")
    p_cli.add_argument("--strength", type=float, default=1.0)
    p_cli.set_defaults(func=run_cli)

    p_video = sub.add_parser("video", parents=[common], help="swap a source identity into a video")
    p_video.add_argument("--source", required=True, help="source identity image")
    p_video.add_argument("--input", required=True, help="input video file")
    p_video.add_argument("--out", required=True, help="output video file")
    p_video.add_argument("--working-size", type=int, default=None, help="generator crop size")
    p_video.add_argument("--detect-every", type=int, default=None, help="re-detect every N frames")
    p_video.set_defaults(func=run_video)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

