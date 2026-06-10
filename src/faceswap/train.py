"""Training loop: data pipeline + frozen encoder + generator + discriminator."""
from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# from written functions
from .data.dataset import FaceSwapDataset, denormalize, load_manifest, worker_init_fn
from .losses import (
    FeatureMatchingLoss,
    HingeGANLoss,
    identity_loss,
    reconstruction_loss,
)
from .models import Generator, IdentityEncoder, MultiscaleDiscriminator
from .utils.device import device_supports_amp, get_device
from .utils.seed import set_seed

log = logging.getLogger(__name__)


#######################################
# Builders
########################################
def build_dataloader(cfg: dict) -> DataLoader:
    data_cfg, train_cfg = cfg["data"], cfg["train"]
    manifest = load_manifest(data_cfg["manifest"])
    dataset = FaceSwapDataset(
        manifest["entries"],
        image_size=data_cfg["image_size"],
        same_id_prob=train_cfg["same_id_prob"],
        augment_flip=True,
    )
    use_cuda = get_device(cfg["run"]["device"]).type == "cuda"
    return DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        worker_init_fn=worker_init_fn,
        pin_memory=use_cuda,
        drop_last=True,
    )


def build_models(cfg: dict, device: torch.device):
    m = cfg["model"]
    encoder = IdentityEncoder(
        backbone=m["id_encoder"]["backbone"],
        weights_path=m["id_encoder"].get("weights"),
    ).to(device)
    if m["id_encoder"].get("weights") is None:
        log.warning(
            "No ArcFace weights provided (model.id_encoder.weights is null): the "
            "identity loss will be MEANINGLESS. Set real weights before a real run."
        )
    generator = Generator(
        base_channels=m["generator_channels"],
        id_embed_dim=m["id_embed_dim"],
        num_id_blocks=m["num_id_blocks"],
        num_scales=m["num_scales"],
    ).to(device)
    discriminator = MultiscaleDiscriminator(
        ndf=m["discriminator"]["base_channels"],
        n_layers=m["discriminator"]["n_layers"],
        num_D=m["discriminator"]["num_scales"],
    ).to(device)
    return encoder, generator, discriminator


#######################################
# Checkpointing
#######################################
def save_checkpoint(path: Path, step: int, generator, discriminator, opt_g, opt_d, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "config": cfg,
        },
        path,
    )
    log.info("checkpoint saved: %s (step %d)", path, step)


def load_checkpoint(path: str | Path, generator, discriminator, opt_g, opt_d,device):
    # map_location so a CUDA checkpoint resumes on MPS or CPU without manual remap. NOT IDEAL due to precision issues!!
    ckpt = torch.load(path, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    discriminator.load_state_dict(ckpt["discriminator"])
    if opt_g is not None and "opt_g" in ckpt:
        opt_g.load_state_dict(ckpt["opt_g"])
    if opt_d is not None and "opt_d" in ckpt:
        opt_d.load_state_dict(ckpt["opt_d"])
    return int(ckpt.get("step", 0))


def save_samples(path: Path, target, source, swapped, max_n: int = 4):
    """Write a [target | source | swapped] grid for visual monitoring."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = min(max_n, target.size(0))
    rows = []
    for i in range(n):
        triple = [denormalize(t[i]).permute(1, 2, 0).cpu().numpy() for t in (target, source, swapped)]
        rows.append(np.concatenate(triple, axis=1))
    grid_rgb = np.concatenate(rows, axis=0)
    cv2.imwrite(str(path), cv2.cvtColor(grid_rgb, cv2.COLOR_RGB2BGR))


#######################################
# Training
#######################################
def train(cfg: dict, resume: Optional[str] = None, max_steps: Optional[int] = None) -> dict:
    """Run training. Returns the last step loss dict for testing"""
    set_seed(cfg["run"]["seed"])
    device = get_device(cfg["run"]["device"])
    use_amp = bool(cfg["train"].get("use_amp", False)) and device_supports_amp(device)
    log.info("device=%s amp=%s", device, use_amp)

    loader = build_dataloader(cfg)
    encoder, generator, discriminator = build_models(cfg, device)

    tc = cfg["train"]
    # betas=(0, 0.999): near-zero first moment standard for image GANs
    opt_g = torch.optim.Adam(generator.parameters(), lr=tc["lr_g"], betas=(0.0, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=tc["lr_d"], betas=(0.0, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_step = 0
    if resume:
        start_step = load_checkpoint(resume, generator, discriminator, opt_g, opt_d, device)
        log.info("resumed from %s at step %d", resume, start_step)

    gan = HingeGANLoss()
    fm_loss = FeatureMatchingLoss(
        weak=tc.get("weak_feature_matching", True), weak_layers=tc.get("weak_fm_layers", 2)
    ).to(device)
    w = tc["loss_weights"]

    total_steps = max_steps if max_steps is not None else tc["total_steps"]
    out_dir = Path(cfg["run"]["output_dir"])

    generator.train()
    discriminator.train()
    data_iter = itertools.cycle(loader)
    last_losses: dict = {}

    def autocast():
        return torch.amp.autocast(device_type=device.type, enabled=use_amp)

    for step in range(start_step, total_steps):
        batch = next(data_iter)
        target = batch["target"].to(device, non_blocking=True)
        source = batch["source"].to(device, non_blocking=True)
        same_mask = batch["same_identity"].to(device)

        with torch.no_grad():
            src_emb = encoder(source)            # source identity, no grad needed

        # Discriminator step
        opt_d.zero_grad(set_to_none=True)
        with autocast():
            fake = generator(target, src_emb).detach()
            d_real = discriminator(target)
            d_fake = discriminator(fake)
            d_loss = gan.d_loss(d_real, d_fake)
        scaler.scale(d_loss).backward()
        scaler.step(opt_d)

        # Generator  step
        opt_g.zero_grad(set_to_none=True)
        with autocast():
            fake = generator(target, src_emb)
            d_fake = discriminator(fake)
            d_real = discriminator(target)
            emb_fake = encoder(fake)             # grad goes through frozen encoder

            adv = gan.g_loss(d_fake)
            idl = identity_loss(emb_fake, src_emb)
            rec = reconstruction_loss(fake, target, same_mask)
            fm = fm_loss(d_fake, d_real)
            g_loss = (
                w["adversarial"] * adv
                + w["identity"] * idl
                + w["reconstruction"] * rec
                + w["feature_matching"] * fm
            )
        scaler.scale(g_loss).backward()
        scaler.step(opt_g)
        scaler.update()

        last_losses = {
            "d": float(d_loss.detach()),
            "g": float(g_loss.detach()),
            "adv": float(adv.detach()),
            "id": float(idl.detach()),
            "rec": float(rec.detach()),
            "fm": float(fm.detach()),
        }

        if (step + 1) % tc["log_every"] == 0:
            log.info(
                "step %d | D %.3f | G %.3f (adv %.3f id %.3f rec %.3f fm %.3f)",
                step + 1, last_losses["d"], last_losses["g"],
                last_losses["adv"], last_losses["id"], last_losses["rec"], last_losses["fm"],
            )
            save_samples(out_dir / "samples" / f"step_{step + 1:07d}.png",
                         target, source, fake.detach())

        if (step + 1) % tc["checkpoint_every"] == 0:
            save_checkpoint(out_dir / "checkpoints" / f"step_{step + 1:07d}.pt",
                            step + 1, generator, discriminator, opt_g, opt_d, cfg)

    return last_losses


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Train the face-swap generator.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--max-steps", type=int, default=None, help="override total_steps")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg, resume=args.resume, max_steps=args.max_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



