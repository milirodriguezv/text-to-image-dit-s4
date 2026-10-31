"""Text-to-image inference for the project's DiT-S/4. No training is performed."""
import argparse
import hashlib
import json
import math
import warnings
from time import perf_counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CLIP_NAME = "openai/clip-vit-large-patch14"
VAE_NAME = "stabilityai/sd-vae-ft-ema"
LATENT_SCALE = 0.18215


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions():
    result = {}
    for name in ("torch", "torchvision", "timm", "diffusers", "transformers", "torchmetrics", "clean-fid"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def runtime_info(device):
    device = torch.device(device)
    return {"hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else ("Apple Silicon MPS" if device.type == "mps" else "cpu"),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "num_threads": torch.get_num_threads(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32}


def resolve_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError("Use cpu, cuda or mps for generation")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable")
    return device


def validate_sampling(sampler, steps, cfg_scale):
    if sampler not in ("dpm", "ddpm"):
        raise ValueError("sampler must be dpm or ddpm")
    if not 2 <= steps <= 1000:
        raise ValueError("steps must be between 2 and 1000")
    if not math.isfinite(cfg_scale) or cfg_scale < 0:
        raise ValueError("CFG must be finite and non-negative")


@torch.inference_mode()
def sample_latents(model, embeddings, null_embedding, *, sampler="dpm", steps=20,
                   cfg_scale=4.0, seed=0):
    """Sample (B,4,32,32); seed and batch size define a reproducible batch.

    DPM uses epsilon only. DDPM retains the learned variance and the repository's
    original timestep remapping. Both guide all four epsilon channels.
    """
    validate_sampling(sampler, steps, cfg_scale)
    if embeddings.ndim != 2 or embeddings.shape[1] != 768 or not len(embeddings):
        raise ValueError("Expected a non-empty (B,768) caption tensor")
    if null_embedding.shape != (1, 768):
        raise ValueError("Expected null embedding (1,768)")
    if not torch.isfinite(embeddings).all() or not torch.isfinite(null_embedding).all():
        raise ValueError("Text embeddings contain non-finite values")
    device, n = embeddings.device, len(embeddings)
    model.eval()
    y = torch.cat([embeddings.float(), null_embedding.to(embeddings).expand(n, -1).float()])
    # Noise is drawn on CPU, so its initial value is independent of the inference device.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(n, 4, 32, 32, generator=generator).to(device)
    if sampler == "dpm":
        from diffusers import DPMSolverMultistepScheduler
        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02,
            beta_schedule="linear", algorithm_type="dpmsolver++", solver_order=2,
            prediction_type="epsilon", thresholding=False, solver_type="midpoint",
            lower_order_final=True, timestep_spacing="linspace", use_karras_sigmas=False,
            final_sigmas_type="zero",
        )
        scheduler.set_timesteps(steps, device=device)
        for t in scheduler.timesteps:
            times = t.expand(2 * n).long()
            output = model(torch.cat([z, z]), times, y)
            cond, uncond = output[:, :4].float().chunk(2)
            eps = uncond + cfg_scale * (cond - uncond)
            z = scheduler.step(eps, t, z).prev_sample
    else:
        from diffusion import create_diffusion
        diffusion = create_diffusion(str(steps))
        if device.type == "mps":
            # _extract_into_tensor ultimately uses fp32, but transfers float64 arrays
            # before its cast. Cast the same inference coefficients before transfer;
            # leave the repository diffusion implementation and training unchanged.
            for name, value in vars(diffusion).items():
                if isinstance(value, np.ndarray) and value.dtype == np.float64:
                    setattr(diffusion, name, value.astype(np.float32))
        # fork_rng prevents inference from changing the caller's training RNG state.
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        if device.type == "mps":
            devices = [0]
        with torch.random.fork_rng(devices=devices, device_type="mps" if device.type == "mps" else "cuda"):
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            elif device.type == "mps":
                torch.mps.manual_seed(seed)
            z = diffusion.p_sample_loop(
                model.forward_with_cfg, (2 * n, 4, 32, 32), torch.cat([z, z]),
                clip_denoised=False, model_kwargs={"y": y, "cfg_scale": cfg_scale},
                progress=False, device=device,
            ).chunk(2)[0]
    if not torch.isfinite(z).all():
        raise RuntimeError("Sampler produced non-finite latents; check checkpoint and configuration")
    return z


@torch.inference_mode()
def decode_latents(vae, latents):
    decoded = vae.decode(latents.float() / LATENT_SCALE).sample
    if not torch.isfinite(decoded).all():
        raise RuntimeError("VAE produced non-finite pixels")
    return ((decoded.float() + 1) / 2).clamp(0, 1)


def save_images(images, paths):
    if len(images) != len(paths) or images.shape[1:] != (3, 256, 256):
        raise ValueError("Expected one path per RGB 256x256 image")
    arrays = (images.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)
    for array, path in zip(arrays, paths):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        Image.fromarray(array).save(temporary, format="PNG")
        temporary.replace(path)


def save_preview(paths, prompts, output):
    """A labelled grid of up to eight samples; stored outside the metric image folder."""
    import textwrap
    from PIL import ImageDraw

    count = min(8, len(paths))
    columns = min(4, count)
    grid = Image.new("RGB", (columns * 256, math.ceil(count / columns) * 296), "white")
    draw = ImageDraw.Draw(grid)
    for i, (path, prompt) in enumerate(zip(paths[:count], prompts[:count])):
        left, top = (i % columns) * 256, (i // columns) * 296
        with Image.open(path) as image:
            grid.paste(image.convert("RGB"), (left, top))
        label = "\n".join(textwrap.wrap(prompt, 38)[:2])
        draw.text((left + 4, top + 258), label, fill="black")
    grid.save(output)


class TextToImagePipeline:
    def __init__(self, model, tokenizer, text_encoder, vae, device, checkpoint_path, training=None):
        self.model = model.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder.eval().requires_grad_(False)
        self.vae = vae.eval().requires_grad_(False)
        self.device = torch.device(device)
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        self.checkpoint_sha256 = sha256(checkpoint_path)
        self.training = training or {}
        self.null_condition = "zeros_from_training_checkpoint" if not torch.count_nonzero(model.y_embedder.null_embedding) else "embedding_from_training_checkpoint"

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device="auto", local_files_only=False, *, allow_zero_null=False):
        from download import find_model
        from models import DiT_models
        from transformers import CLIPTokenizer, CLIPTextModel
        from diffusers import AutoencoderKL

        device = resolve_device(device)
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        initial_hash = sha256(path)
        state, training = find_model(str(path), return_metadata=True)
        if not isinstance(state, dict) or "y_embedder.proj.0.weight" not in state:
            raise ValueError("Expected EMA weights for text-conditioned DiT-S/4, not original class weights")
        null = state.get("y_embedder.null_embedding")
        if null is None or null.shape != (1, 768) or not torch.isfinite(null).all():
            raise ValueError("Checkpoint must contain a finite (1,768) null embedding")
        if not torch.count_nonzero(null):
            if not allow_zero_null:
                raise ValueError("Checkpoint null embedding is zero, not the planned nonzero CLIP empty-text embedding. "
                                 "Use --allow-zero-null only to evaluate the original zero-null training convention.")
            warnings.warn("Preserving the zero null embedding used in training. This differs from the planned CLIP empty-text condition.", UserWarning)
        args = training.get("args", {})
        if args.get("model", "DiT-S/4") != "DiT-S/4" or args.get("image_size", 256) != 256:
            raise ValueError("Checkpoint training configuration is not DiT-S/4 at image size 256")
        model = DiT_models["DiT-S/4"](input_size=32, text_dim=768, null_path=None)
        model.load_state_dict(state, strict=True)
        model.to(device)
        options = {"local_files_only": local_files_only}
        tokenizer = CLIPTokenizer.from_pretrained(CLIP_NAME, **options)
        text_encoder = CLIPTextModel.from_pretrained(CLIP_NAME, **options).to(device)
        vae = AutoencoderKL.from_pretrained(VAE_NAME, **options).to(device)
        pipeline = cls(model, tokenizer, text_encoder, vae, device, path, training)
        if pipeline.checkpoint_sha256 != initial_hash:
            raise RuntimeError("Checkpoint changed while loading; use an immutable copy")
        return pipeline

    @torch.inference_mode()
    def generate(self, prompts, *, sampler="dpm", steps=None, cfg_scale=4.0, seed=0):
        if not prompts or any(not isinstance(p, str) for p in prompts):
            raise ValueError("Expected a non-empty list of text prompts")
        steps = (20 if sampler == "dpm" else 250) if steps is None else steps
        tokens = self.tokenizer(prompts, padding="max_length", max_length=77,
                                truncation=True, return_tensors="pt").to(self.device)
        e = self.text_encoder(**tokens).pooler_output.float()
        z = sample_latents(self.model, e, self.model.y_embedder.null_embedding,
                           sampler=sampler, steps=steps, cfg_scale=cfg_scale, seed=seed)
        return decode_latents(self.vae, z)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Trusted local training checkpoint containing ema")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", action="append", help="Repeat for multiple prompts")
    source.add_argument("--prompts-file", type=Path, help="UTF-8, one prompt per line")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--sampler", choices=["dpm", "ddpm"], default="dpm")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-zero-null", action="store_true", help="Preserve a zero null buffer used in the original training run")
    args = parser.parse_args()
    prompts = args.prompt if args.prompt is not None else args.prompts_file.read_text(encoding="utf-8").splitlines()
    if args.batch_size < 1 or not prompts:
        parser.error("A positive batch size and at least one prompt are required")
    steps = args.steps if args.steps is not None else (20 if args.sampler == "dpm" else 250)
    validate_sampling(args.sampler, steps, args.cfg_scale)
    if args.output.exists():
        parser.error("Output directory already exists; use a new directory")
    load_start = perf_counter()
    pipe = TextToImagePipeline.from_checkpoint(args.ckpt, args.device, args.local_files_only,
                                               allow_zero_null=args.allow_zero_null)
    load_seconds = perf_counter() - load_start
    args.output.mkdir(parents=True)
    records = []
    generation_start = perf_counter()
    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start:start + args.batch_size]
        seed = args.seed + start
        images = pipe.generate(batch, sampler=args.sampler, steps=steps, cfg_scale=args.cfg_scale, seed=seed)
        paths = [args.output / f"{i:06d}.png" for i in range(start, start + len(batch))]
        save_images(images, paths)
        records.extend({"prompt": p, "image": path.name, "batch_seed": seed} for p, path in zip(batch, paths))
    generation_seconds = perf_counter() - generation_start
    metadata = {"checkpoint": pipe.checkpoint_path, "checkpoint_sha256": pipe.checkpoint_sha256,
                "sampler": args.sampler, "steps": steps, "cfg_scale": args.cfg_scale,
                "seed": args.seed, "batch_size": args.batch_size, "device": str(pipe.device),
                "clip": CLIP_NAME, "vae": VAE_NAME, "weights": "ema", "versions": package_versions(),
                "runtime": runtime_info(pipe.device),
                "training": pipe.training, "null_condition": pipe.null_condition,
                "timing": {"model_load_seconds": load_seconds,
                           "generation_and_save_seconds": generation_seconds,
                           "images_per_second": len(records) / generation_seconds},
                "samples": records}
    (args.output / "samples.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    save_preview([args.output / row["image"] for row in records], prompts, args.output / "preview.png")
    print(f"Saved {len(records)} images to {args.output}")


if __name__ == "__main__":
    main()
