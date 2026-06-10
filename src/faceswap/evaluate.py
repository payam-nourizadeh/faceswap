"""]
Evaluation orchestrator: metric suite, Pareto sweep, JSON report, graphs.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import torch
import yaml

# from written functions
from .data.dataset import denormalize
from .evaluation import (
    cosine_similarity,
    identity_preservation,
    interpolate_identity,
    plots,
    pose_error,
)
from .evaluation import metrics as M
from .models import Generator, IdentityEncoder
from .utils.device import get_device

log = logging.getLogger(__name__)


@dataclass
class SweepPoint:
    strength: float
    identity_similarity: float
    attribute_error: float


class Evaluator:
    """Runs swap generation, the metric suite, and the Pareto sweep.
    Two identity encoders with different roles:
      gen_encoder:  training backbone (r50)- conditions generation.
      eval_encoder: held-out backbone ( r100) - scores identity only.
    """

    def __init__(self, generator: torch.nn.Module, gen_encoder: torch.nn.Module, eval_encoder: torch.nn.Module, 
                 detect_landmarks: Callable[[np.ndarray], Optional[np.ndarray]], image_size: int, device: torch.device,):
        self.generator = generator.eval()
        self.gen_encoder = gen_encoder.eval()
        self.eval_encoder = eval_encoder.eval()
        self.detect_landmarks = detect_landmarks
        self.image_size = image_size
        self.device = device

    @torch.no_grad()
    def embed_for_generation(self, images: torch.Tensor) -> np.ndarray:
        #training-backbone embedding; used to for generator
        return self.gen_encoder(images.to(self.device)).cpu().numpy()

    @torch.no_grad()
    def embed(self, images: torch.Tensor) -> np.ndarray:
        #Held-out backbone embedding; used to for score identity
        return self.eval_encoder(images.to(self.device)).cpu().numpy()

    @torch.no_grad()
    def swap(self, target: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        return self.generator(target.to(self.device), identity.to(self.device))

    def _landmarks_of(self, image_tensor: torch.Tensor) -> Optional[np.ndarray]:
        #Detect 5-point landmarks on a single CHW [-1,1] image tensor
        import cv2

        rgb = (denormalize(image_tensor).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        # print(np.max(rgb))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return self.detect_landmarks(bgr)

    @torch.no_grad()
    def score_identity(self, swaps: torch.Tensor, sources: torch.Tensor, impostor_scores: np.ndarray,far_target: float = 0.001, ) -> M.IdentityReport:
        #Identity consistency swaps vs sources, threshold calibrated at far_target
        swap_emb = self.embed(swaps)
        src_emb = self.embed(sources)
        sims = cosine_similarity(swap_emb, src_emb)
        return identity_preservation(sims, impostor_scores, far_target)

    def pareto_sweep(
        self, target: torch.Tensor, source: torch.Tensor, strengths: list[float]
    ) -> list[SweepPoint]:
        #sweep identity injection strength, measuring identity sim vs pose error """
        # Generation embeddings are in training space; scoring in held-out space.
        gen_src = self.embed_for_generation(source)
        gen_tgt = self.embed_for_generation(target)
        score_src = self.embed(source)
        points: list[SweepPoint] = []

        for alpha in strengths:
            blended = interpolate_identity(gen_tgt, gen_src, alpha)
            swapped = self.swap(target, torch.from_numpy(blended).float())
            swap_score = self.embed(swapped)

            id_sim = float(cosine_similarity(swap_score, score_src).mean())

            errors = []
            for i in range(target.size(0)):
                t_lmk = self._landmarks_of(target[i])
                s_lmk = self._landmarks_of(swapped[i])
                if t_lmk is not None and s_lmk is not None:
                    errors.append(pose_error(t_lmk, s_lmk, self.image_size)["total_error"])
            attr_err = float(np.nanmean(errors)) if errors else float("nan")

            points.append(SweepPoint(alpha, id_sim, attr_err))
        return points


#######################################
# Report assembly to be testable
#######################################
def build_report(identity: M.IdentityReport, sweep: list[SweepPoint], fid: Optional[float], extra: Optional[dict] = None) -> dict:
    report = {
        "identity": asdict(identity),
        "fid": fid,
        "pareto_sweep": [asdict(p) for p in sweep],
    }
    if extra:
        report.update(extra)
    return report


def render_graphs(report: dict, out_dir: str | Path, history: Optional[dict] = None) -> dict[str, Path]:
    # accuracy graphs from a computed report. returns output paths
    out_dir = Path(out_dir)
    paths: dict[str, Path] = {}

    sweep = report["pareto_sweep"]
    paths["pareto"] = plots.plot_pareto(
        [p["strength"] for p in sweep],
        [p["identity_similarity"] for p in sweep],
        [p["attribute_error"] for p in sweep],
        out_dir / "pareto.png",
    )

    if history:
        steps = list(range(1, len(next(iter(history.values()))) + 1))
        paths["loss"] = plots.plot_loss_curves(history, steps, out_dir / "loss_curves.png")

    if "identity_scores" in report and "impostor_scores" in report:
        paths["identity"] = plots.plot_identity_histogram(
            np.asarray(report["identity_scores"]),
            np.asarray(report["impostor_scores"]),
            threshold=report["identity"]["threshold"],
            far_target=report["identity"]["far_target"],
            out_path=out_dir / "identity_histogram.png",
        )

    if "comparison" in report:
        paths["comparison"] = plots.plot_comparison_bars(
            report["comparison"], out_dir / "comparison.png"
        )
    return paths


def _impostor_scores(evaluator: "Evaluator", sources: "torch.Tensor") -> np.ndarray:
    # Off-diagonal pairs of source embeddings = cross-identity = impostors.
    # Used to calibrate the verification threshold.
    emb = evaluator.embed(sources)
    n = emb.shape[0]
    sims = emb @ emb.T
    iu = np.triu_indices(n, k=1) #upper trainagle part of array
    return sims[iu]


def run_evaluation(evaluator: "Evaluator", loader, far_target: float, strengths: list[float], out_dir,
                    max_batches: Optional[int] = None, fid_fn: Optional[Callable] = None, history: Optional[dict] = None,) -> dict:
    """Iterate cross-identity pairs, compute metrics, write report + graphs.

    fid_fn: optional callable(real_tensor, fake_tensor) -> float. None skips FID
    TO DO: needs InceptionV3 download; the rest of the report works
    """
    import torch

    out_dir = Path(out_dir)
    swap_sims: list[np.ndarray] = []
    impostor: list[np.ndarray] = []
    pose_errors: list[float] = []
    first_sweep: Optional[list[SweepPoint]] = None
    reals, fakes = [], []

    for b, batch in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        target, source = batch["target"], batch["source"]

        # Full-strength swap (alpha=1) for headline identity/attribute numbers.
        gen_src = evaluator.embed_for_generation(source)
        swapped = evaluator.swap(target, torch.from_numpy(gen_src).float())

        swap_sims.append(cosine_similarity(evaluator.embed(swapped), evaluator.embed(source)))
        impostor.append(_impostor_scores(evaluator, source))

        for i in range(target.size(0)):
            t_lmk = evaluator._landmarks_of(target[i])
            s_lmk = evaluator._landmarks_of(swapped[i])
            if t_lmk is not None and s_lmk is not None:
                pose_errors.append(pose_error(t_lmk, s_lmk, evaluator.image_size)["total_error"])

        if fid_fn is not None:
            reals.append(target.cpu())
            fakes.append(swapped.cpu())

        if first_sweep is None or all(np.isnan(p.attribute_error) for p in first_sweep):
            first_sweep = evaluator.pareto_sweep(target, source, strengths)

    swap_sims = np.concatenate(swap_sims)
    impostor = np.concatenate(impostor)
    identity = identity_preservation(swap_sims, impostor, far_target)
    fid = None
    if fid_fn is not None and reals:
        fid = float(fid_fn(torch.cat(reals), torch.cat(fakes)))

    report = build_report(
        identity, first_sweep or [], fid,
        extra={
            "attribute_pose_error_mean": float(np.mean(pose_errors)) if pose_errors else None,
            "identity_scores": swap_sims.tolist(),
            "impostor_scores": impostor.tolist(),
            "num_pairs": int(swap_sims.shape[0]),
        },
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    render_graphs(report, out_dir, history=history)
    return report


#######################################
# from config + CLI
#######################################
def build_evaluator(cfg: dict, checkpoint: str, device: torch.device) -> Evaluator:
    m = cfg["model"]
    ev = cfg["eval"]

    train_backbone = m["id_encoder"]["backbone"]
    eval_backbone = ev["id_eval_backbone"]
    if eval_backbone == train_backbone:
        raise ValueError(
            f"eval backbone ({eval_backbone}) must DIFFER from the training "
            f"backbone ({train_backbone}); otherwise identity scores measure how "
            f"well the generator gamed its own loss network, not identity transfer."
        )

    generator = Generator(
        base_channels=m["generator_channels"],
        id_embed_dim=m["id_embed_dim"],
        num_id_blocks=m["num_id_blocks"],
        num_scales=m["num_scales"],
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    generator.load_state_dict(ckpt["generator"])

    # gen_encoder must match the training backbone — different embedding space
    # would produce garbage swaps. eval_encoder is the held-out scoring network.
    gen_encoder = IdentityEncoder(
        backbone=train_backbone, weights_path=m["id_encoder"].get("weights")
    ).to(device)
    eval_encoder = IdentityEncoder(
        backbone=eval_backbone, weights_path=ev.get("id_eval_weights")
    ).to(device)

    from .data.alignment import InsightFaceDetector

    detector = InsightFaceDetector()

    def detect(image_bgr):
        faces = detector.detect(image_bgr)
        if not faces:
            return None
        return np.asarray(faces[0]["kps"], dtype=np.float32)

    return Evaluator(
        generator, gen_encoder, eval_encoder, detect, cfg["data"]["image_size"], device
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Evaluate a trained face-swap model.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="runs/eval")
    ap.add_argument("--max-batches", type=int, default=50,
                    help="cap evaluation batches (full run can be large)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = get_device(cfg["run"]["device"])
    evaluator = build_evaluator(cfg, args.checkpoint, device)

    # Cross-identity pairs only (same_id_prob=0).
    from torch.utils.data import DataLoader

    from .data.dataset import FaceSwapDataset, load_manifest

    manifest = load_manifest(cfg["data"]["manifest"])
    dataset = FaceSwapDataset(
        manifest["entries"], image_size=cfg["data"]["image_size"],
        same_id_prob=0.0, augment_flip=False,
    )
    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=True)

    report = run_evaluation(
        evaluator, loader,
        far_target=cfg["eval"]["far_target"],
        strengths=cfg["eval"]["strengths"],
        out_dir=args.out,
        max_batches=args.max_batches,
        fid_fn=None,   # set an InceptionV3-based FID callable to include realism
    )
    log.info(
        "identity preservation @ FAR %.3f: %.1f%% over %d pairs -> %s/report.json",
        report["identity"]["far_target"],
        100 * report["identity"]["preservation_rate"],
        report["num_pairs"], args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




