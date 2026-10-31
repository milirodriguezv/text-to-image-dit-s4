# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Training script for text-conditioned DiT-S/4 (adaLN-Zero-Text).
Adapted from the original class-conditional DiT training script.

Single-GPU (or CPU) training using precomputed VAE latents and CLIP embeddings.

Usage:
  python train.py --feature-path features/train --results-dir results \
      --model DiT-S/4 --max-steps 400000

Resume:
  python train.py --feature-path features/train --results-dir results \
      --model DiT-S/4 --max-steps 400000 --resume results/000-DiT-S-4/checkpoints/last.pt
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import os
# CPU optimization: set thread count before any computation.
# 88 threads is optimal on dual-socket EPYC 9474F (avoids NUMA contention at 96).
_CPU_THREADS = int(os.environ.get("OMP_NUM_THREADS", 88))
torch.set_num_threads(_CPU_THREADS)

import numpy as np
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
from pathlib import Path
import argparse
import logging
import os
import sys

from models import DiT_models
from diffusion import create_diffusion
from dataset_t2i import CocoLatentDataset


# ---------------------------------------------------------------------------
#  Training helper functions (kept from original)
# ---------------------------------------------------------------------------

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Step the EMA model towards the current model."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """Set requires_grad flag for all parameters in a model."""
    for p in model.parameters():
        p.requires_grad = flag


# ---------------------------------------------------------------------------
#  bf16 wrapper — autocast only the model forward, not the whole
#  training_losses call (which computes the loss in fp32).
# ---------------------------------------------------------------------------

class AutocastModelWrapper:
    """Thin callable that runs model(x, t, **kw) under autocast and casts
    the output back to float32.  Passed to diffusion.training_losses() so
    that the loss computation itself stays in fp32."""

    def __init__(self, model, device_type, enabled=True):
        self.model = model
        self.device_type = device_type
        self.enabled = enabled

    def __call__(self, x, t, **kwargs):
        with torch.autocast(self.device_type, dtype=torch.bfloat16, enabled=self.enabled):
            out = self.model(x, t, **kwargs)
        return out.float()


# ---------------------------------------------------------------------------
#  Periodic sampling with DPM-Solver++
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_samples(model, vae, null_emb, clip_tokenizer, clip_model,
                     prompts, device, cfg_scale=4.0, seed=0):
    """Generate a grid of images from text prompts using DPM-Solver++ (20 steps).
    Returns a (B, 3, 256, 256) tensor in [0, 1]."""
    from diffusers import DPMSolverMultistepScheduler
    from text_utils import encode_texts

    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=0.02,
        algorithm_type="dpmsolver++",
        solver_order=2,
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(20, device=device)

    n = len(prompts)
    generator = torch.Generator(device=device).manual_seed(seed)

    # Encode prompts
    e_prompts = encode_texts(prompts, clip_tokenizer, clip_model, device)  # (n, 768)
    e_null = null_emb.expand(n, -1).to(device)  # (n, 768)
    y = torch.cat([e_prompts, e_null], dim=0)  # (2n, 768)

    # Initial noise
    z = torch.randn(n, 4, 32, 32, device=device, generator=generator)

    model.eval()
    for t_val in scheduler.timesteps:
        t_batch = t_val.expand(2 * n).long().to(device)
        z_doubled = torch.cat([z, z], dim=0)
        model_out = model(z_doubled, t_batch, y=y)
        eps, sigma_channels = model_out[:, :4], model_out[:, 4:]
        cond_eps, uncond_eps = eps.chunk(2)
        guided_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        # Only use the guided eps for the scheduler step (ignore sigma channels)
        z = scheduler.step(guided_eps, t_val, z).prev_sample

    # Decode latents
    z = z / 0.18215
    images = vae.decode(z).sample
    images = ((images + 1) / 2).clamp(0, 1)
    return images


# ---------------------------------------------------------------------------
#  Logger
# ---------------------------------------------------------------------------

def create_logger(logging_dir):
    """Create a logger that writes to a log file and stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    if logging_dir is not None:
        logger.addHandler(logging.FileHandler(f"{logging_dir}/log.txt"))
    return logger


# ---------------------------------------------------------------------------
#  Main training loop
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.type == "cuda" else "cpu"
    # bf16: auto-detect AVX-512 BF16 on CPU (3x speedup on EPYC 9474F)
    if args.use_amp == "auto":
        if device.type == "cuda":
            use_amp = True
        else:
            # Check for hardware bf16 via CPU flags
            try:
                use_amp = "avx512_bf16" in open("/proc/cpuinfo").read()
            except OSError:
                use_amp = False
    else:
        use_amp = args.use_amp == "on"

    # ---- Experiment directory ----
    if args.resume:
        # When resuming, reuse the experiment directory from the checkpoint path
        # e.g. results/000-DiT-S-4/checkpoints/last.pt → results/000-DiT-S-4
        experiment_dir = str(Path(args.resume).parent.parent)
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        sample_dir = f"{experiment_dir}/samples"
    else:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        sample_dir = f"{experiment_dir}/samples"
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)

    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Device: {device}, AMP (bf16): {use_amp}")

    # ---- Seed ----
    torch.manual_seed(args.global_seed)
    np.random.seed(args.global_seed)

    # ---- Model ----
    assert args.image_size % 8 == 0
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        text_dim=768,
        class_dropout_prob=args.cfg_dropout,
    ).to(device)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    diffusion = create_diffusion(timestep_respacing="")  # 1000 steps, linear
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Optimizer ----
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    # ---- Resume ----
    train_steps = 0
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        train_steps = ckpt.get("step", 0)
        logger.info(f"Resumed at step {train_steps}")

    # ---- Dataset ----
    dataset = CocoLatentDataset(feature_dir=args.feature_path)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.global_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    logger.info(f"Dataset: {len(dataset):,} images from {args.feature_path}")

    # ---- Prepare for training ----
    update_ema(ema, model, decay=0)  # sync EMA with model
    model.train()
    ema.eval()

    # bf16 wrapper
    amp_wrapper = AutocastModelWrapper(model, device_type, enabled=use_amp)

    # ---- Logging variables ----
    running_loss = 0.0
    running_mse = 0.0
    running_vb = 0.0
    log_steps = 0
    start_time = time()

    # ---- Fixed prompts for periodic sampling ----
    sample_prompts = [
        "A dog sitting on a beach at sunset",
        "A red car parked in front of a building",
        "A woman riding a bicycle in a park",
        "A plate of food on a wooden table",
        "A cat sleeping on a couch",
        "A group of people playing baseball",
        "An airplane flying over mountains",
        "A vase of flowers next to a window",
    ]

    logger.info(f"Training for {args.max_steps} steps (batch {args.global_batch_size}, "
                f"lr {args.lr}, cfg_dropout {args.cfg_dropout})...")

    # ---- Training loop (step-based, not epoch-based) ----
    while train_steps < args.max_steps:
        for z_0, e_pooled in loader:
            if train_steps >= args.max_steps:
                break

            z_0 = z_0.to(device)
            e_pooled = e_pooled.to(device)

            t = torch.randint(0, diffusion.num_timesteps, (z_0.shape[0],), device=device)
            model_kwargs = dict(y=e_pooled)
            loss_dict = diffusion.training_losses(amp_wrapper, z_0, t, model_kwargs)
            loss = loss_dict["loss"].mean()

            opt.zero_grad()
            loss.backward()

            # Log gradient norm (no clipping, as in the paper)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))

            opt.step()
            update_ema(ema, model)

            # Accumulate for logging
            running_loss += loss.item()
            running_mse += loss_dict["mse"].mean().item()
            running_vb += loss_dict["vb"].mean().item()
            log_steps += 1
            train_steps += 1

            # ---- Logging ----
            if train_steps % args.log_every == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end_time = time()
                secs = end_time - start_time
                steps_per_sec = log_steps / secs
                avg_loss = running_loss / log_steps
                avg_mse = running_mse / log_steps
                avg_vb = running_vb / log_steps
                logger.info(
                    f"(step={train_steps:07d}) loss={avg_loss:.4f}  "
                    f"mse={avg_mse:.4f}  vb={avg_vb:.6f}  "
                    f"|grad|={grad_norm:.4f}  "
                    f"steps/s={steps_per_sec:.2f}"
                )
                running_loss = 0.0
                running_mse = 0.0
                running_vb = 0.0
                log_steps = 0
                start_time = time()

            # ---- Checkpointing ----
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "step": train_steps,
                    "args": vars(args),
                }
                # Overwrite last.pt
                last_path = f"{checkpoint_dir}/last.pt"
                torch.save(checkpoint, last_path)
                logger.info(f"Saved {last_path} (step {train_steps})")

            if train_steps % args.permanent_ckpt_every == 0 and train_steps > 0:
                perm_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "step": train_steps,
                    "args": vars(args),
                }
                torch.save(checkpoint, perm_path)
                logger.info(f"Saved permanent checkpoint {perm_path}")

            # ---- Periodic sampling ----
            if train_steps % args.sample_every == 0 and train_steps > 0:
                try:
                    from text_utils import load_clip
                    from diffusers.models import AutoencoderKL
                    from torchvision.utils import make_grid, save_image

                    logger.info(f"Generating samples at step {train_steps}...")
                    clip_tokenizer, clip_model_enc = load_clip(device)
                    vae = AutoencoderKL.from_pretrained(
                        "stabilityai/sd-vae-ft-ema"
                    ).to(device)
                    vae.eval()

                    null_emb = ema.y_embedder.null_embedding  # (1, 768)

                    images = generate_samples(
                        ema, vae, null_emb, clip_tokenizer, clip_model_enc,
                        sample_prompts, device, cfg_scale=4.0, seed=0,
                    )
                    grid = make_grid(images, nrow=4)
                    save_path = f"{sample_dir}/{train_steps:07d}.png"
                    save_image(grid, save_path)
                    logger.info(f"Saved samples to {save_path}")

                    # Free VRAM
                    del vae, clip_tokenizer, clip_model_enc
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"Sample generation failed: {e}")

                model.train()  # re-enable training mode (dropout)

    logger.info(f"Training complete at step {train_steps}.")

    # Save final checkpoint
    checkpoint = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
        "step": train_steps,
        "args": vars(args),
    }
    final_path = f"{checkpoint_dir}/last.pt"
    torch.save(checkpoint, final_path)
    logger.info(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=str, required=True,
                        help="Path to precomputed features directory")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/4")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=400_000)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cfg-dropout", type=float, default=0.15,
                        help="Text dropout probability for CFG (default 0.15)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10_000,
                        help="Save last.pt every N steps")
    parser.add_argument("--permanent-ckpt-every", type=int, default=50_000,
                        help="Save permanent checkpoint every N steps")
    parser.add_argument("--sample-every", type=int, default=10_000,
                        help="Generate samples every N steps")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--use-amp", type=str, default="auto", choices=["auto", "on", "off"],
                        help="bf16 autocast: auto detects AVX-512 BF16 / GPU (default: auto)")
    args = parser.parse_args()
    main(args)
