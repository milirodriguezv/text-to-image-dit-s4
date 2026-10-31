"""Generate evaluation images, compute metrics, or run a complete CFG sweep."""
import argparse
import gc
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from eval.common import checked_image, load_manifest, read_json, write_json
from sample_t2i import (CLIP_NAME, VAE_NAME, TextToImagePipeline, package_versions,
                        resolve_device, runtime_info, save_images, save_preview, sha256, validate_sampling)

CFG_SCALES = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0)


def code_fingerprint():
    """Detect code changes before resuming a partially generated run."""
    import hashlib
    root = Path(__file__).resolve().parents[1]
    paths = [root / name for name in ("sample_t2i.py", "models.py", "download.py")]
    paths += sorted((root / "diffusion").glob("*.py")) + sorted((root / "eval").glob("*.py"))
    return hashlib.sha256("".join(sha256(p) for p in paths).encode()).hexdigest()


def generate_run(pipe, manifest_path, output, *, count=None, sampler="dpm", steps=None,
                 cfg_scale=4.0, seed=0, batch_size=8, resume=False, checkpoint_step=None, progress_callback=None):
    steps = (20 if sampler == "dpm" else 250) if steps is None else steps
    validate_sampling(sampler, steps, cfg_scale)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    manifest_path, output = Path(manifest_path).resolve(), Path(output)
    data, records = load_manifest(manifest_path, count)
    training = pipe.training
    saved_step = training.get("step")
    if checkpoint_step is not None and saved_step is not None and checkpoint_step != saved_step:
        raise ValueError("--checkpoint-step does not match the step stored in the checkpoint")
    checkpoint_step = saved_step if saved_step is not None else checkpoint_step
    config = {"manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
              "checkpoint": pipe.checkpoint_path, "checkpoint_sha256": pipe.checkpoint_sha256,
              "checkpoint_step": checkpoint_step, "model": "DiT-S/4", "weights": "ema",
              "count": len(records), "preprocess": data["preprocess"], "sampler": sampler,
              "steps": steps, "cfg_scale": cfg_scale, "seed": seed, "batch_size": batch_size,
              "device": str(pipe.device), "clip": CLIP_NAME, "vae": VAE_NAME,
              "versions": package_versions(), "code_sha256": code_fingerprint(),
              "runtime": runtime_info(pipe.device), "training": training,
              "null_condition": pipe.null_condition}
    run_path = output / "run.json"
    if output.exists():
        if not resume or not run_path.is_file():
            raise FileExistsError(f"Use a new output directory or --resume for an existing run: {output}")
        previous = read_json(run_path)
        if previous["config"] != config:
            raise ValueError("Cannot resume: checkpoint, manifest, code, environment or sampling configuration changed")
    else:
        output.mkdir(parents=True)
        previous = {"schema": 1, "status": "incomplete", "config": config, "records": []}
        write_json(run_path, previous)
    image_dir = output / "images"
    image_dir.mkdir(exist_ok=True)
    expected = {row["file"] for row in records}
    # A process can stop after writing a PNG temporary file and before its rename.
    for name in expected:
        image_dir.joinpath(name).with_suffix(".tmp").unlink(missing_ok=True)
    if {p.name for p in image_dir.iterdir()} - expected:
        raise ValueError("Unexpected files in generated image directory")
    completed = {row["file"]: row for row in previous["records"]}
    generated = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        paths = [image_dir / row["file"] for row in batch]
        batch_seed = seed + start
        if all(row["file"] in completed for row in batch):
            for path in paths:
                checked_image(path, completed[path.name]["sha256"])
        else:
            # Repeat a whole interrupted batch using exactly the same seed.
            images = pipe.generate([row["caption"] for row in batch], sampler=sampler, steps=steps,
                                   cfg_scale=cfg_scale, seed=batch_seed)
            save_images(images, paths)
        generated.extend({"image_id": row["image_id"], "file": row["file"],
                          "caption": row["caption"], "batch_seed": batch_seed,
                          "sha256": sha256(path)} for row, path in zip(batch, paths))
        write_json(run_path, {"schema": 1, "status": "incomplete", "config": config, "records": generated})
        if progress_callback is not None:
            progress_callback(len(generated), len(records))
        print(f"CFG={cfg_scale:g}: {len(generated)}/{len(records)}", flush=True)
    write_json(run_path, {"schema": 1, "status": "complete", "config": config, "records": generated})
    save_preview([image_dir / row["file"] for row in records], [row["caption"] for row in records], output / "preview.png")
    return run_path


def validate_run(run_path):
    run_path = Path(run_path).resolve()
    run = read_json(run_path)
    if run.get("schema") != 1 or run.get("status") != "complete":
        raise ValueError("Generation is incomplete; resume it before computing metrics")
    config = run["config"]
    if sha256(config["manifest"]) != config["manifest_sha256"]:
        raise ValueError("Evaluation manifest has changed")
    _, records = load_manifest(config["manifest"], config["count"])
    if len(records) != len(run["records"]):
        raise ValueError("Generated count does not match references")
    expected = {row["file"] for row in records}
    actual = {p.name for p in (run_path.parent / "images").iterdir()}
    if actual != expected:
        raise ValueError("Generated directory must contain exactly the manifest image set")
    for real, generated in zip(records, run["records"]):
        if any(real[key] != generated[key] for key in ("file", "image_id", "caption")):
            raise ValueError("Generated images/captions are not aligned with the reference manifest")
        checked_image(run_path.parent / "images" / generated["file"], generated["sha256"])
    return run, records


def compute_metrics(run_path, *, device="auto", batch_size=32, clip_device=None):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    run_path = Path(run_path).resolve()
    run, records = validate_run(run_path)
    device = resolve_device(device)
    # python.org macOS installations may lack an OS CA bundle for urllib downloads.
    # Preserve an explicit user bundle; otherwise use certifi without disabling TLS.
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    from cleanfid import fid

    # A 2K run may use a prefix of a frozen 10K manifest: clean-fid must only see
    # those matching 2K real images, never all 10K references.
    reference_dir = Path(run["config"]["manifest"]).parent / "real"
    image_dir = run_path.parent / "images"
    with tempfile.TemporaryDirectory(prefix="dit-fid-") as temporary:
        for row in records:
            source, target = reference_dir / row["file"], Path(temporary) / row["file"]
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        fid_score = float(fid.compute_fid(str(image_dir), temporary, mode="clean",
                                         model_name="inception_v3", num_workers=0,
                                         batch_size=batch_size, device=device, use_dataparallel=False))
    if device.type == "mps":
        torch.mps.empty_cache()
    scores = compute_clip_scores(image_dir, records, device=clip_device or device, batch_size=batch_size)
    if not all(math.isfinite(value) for value in (fid_score, *scores.values())):
        raise RuntimeError("Non-finite evaluation metric; results were not saved")
    result = {"schema": 1, "run": str(run_path), "run_sha256": sha256(run_path),
              "config": run["config"], "fid": fid_score, "fid_label": f"FID-{len(records)}",
              "fid_implementation": "clean-fid/clean/inception_v3", **scores,
              "metric_device": str(device), "metric_batch_size": batch_size,
              "clip_device": str(resolve_device(clip_device or device)),
              "metric_versions": package_versions()}
    destination = run_path.parent / "metrics.json"
    write_json(destination, result)
    return destination


def compute_clip_scores(image_dir, records, *, device="auto", batch_size=32):
    from torchmetrics.multimodal import CLIPScore
    device = resolve_device(device)
    image_dir = Path(image_dir)
    if batch_size < 1 or not records:
        raise ValueError("CLIP scoring requires a positive batch size and non-empty records")
    scores = {}
    for label, model_name in (("clip_l14", CLIP_NAME), ("clip_b32", "openai/clip-vit-base-patch32")):
        metric = CLIPScore(model_name_or_path=model_name).to(device)
        with torch.inference_mode():
            for start in range(0, len(records), batch_size):
                batch = records[start:start + batch_size]
                arrays = []
                for row in batch:
                    with Image.open(image_dir / row["file"]) as image:
                        arrays.append(np.array(image, dtype=np.uint8))
                images = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(device)
                metric.update(images, [row["caption"] for row in batch])
            scores[label] = float(metric.compute().cpu())
        del metric
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
    if not all(math.isfinite(value) for value in scores.values()):
        raise RuntimeError("Non-finite CLIP Score")
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "sweep"):
        sub = commands.add_parser(command)
        sub.add_argument("--ckpt", required=True)
        sub.add_argument("--manifest", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--count", type=int, help="Prefix of manifest; 2000 for intermediate FID")
        sub.add_argument("--sampler", choices=["dpm", "ddpm"], default="dpm")
        sub.add_argument("--steps", type=int)
        sub.add_argument("--seed", type=int, default=0)
        sub.add_argument("--batch-size", type=int, default=8)
        sub.add_argument("--device", default="auto")
        sub.add_argument("--resume", action="store_true")
        sub.add_argument("--checkpoint-step", type=int, help="Training step for convergence plots")
        sub.add_argument("--local-files-only", action="store_true")
        sub.add_argument("--allow-zero-null", action="store_true", help="Preserve the original zero-null training convention")
        if command == "generate":
            sub.add_argument("--cfg-scale", type=float, default=4.0)
        else:
            sub.add_argument("--cfg-scales", type=float, nargs="+", default=list(CFG_SCALES))
            sub.add_argument("--metrics", action="store_true", help="Compute metrics and plots after all generation")
            sub.add_argument("--metric-batch-size", type=int, default=32)
    sub = commands.add_parser("metrics")
    sub.add_argument("--run", type=Path, required=True, help="run.json from completed generation")
    sub.add_argument("--device", default="auto")
    sub.add_argument("--batch-size", type=int, default=32)
    sub.add_argument("--clip-device", help="Separate CLIP device, e.g. mps while FID uses cpu")
    args = parser.parse_args()
    if args.command == "metrics":
        print(compute_metrics(args.run, device=args.device, batch_size=args.batch_size, clip_device=args.clip_device))
        return
    load_manifest(args.manifest, args.count)  # validate before loading GPU models
    scales = [args.cfg_scale] if args.command == "generate" else args.cfg_scales
    if len(set(scales)) != len(scales):
        parser.error("CFG scales must be unique")
    for scale in scales:
        validate_sampling(args.sampler, args.steps if args.steps is not None else (20 if args.sampler == "dpm" else 250), scale)
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    pipe = TextToImagePipeline.from_checkpoint(args.ckpt, args.device, args.local_files_only,
                                               allow_zero_null=args.allow_zero_null)
    paths = []
    for scale in scales:
        output = args.output if args.command == "generate" else args.output / f"cfg_{scale:g}"
        paths.append(generate_run(pipe, args.manifest, output, count=args.count, sampler=args.sampler,
                                  steps=args.steps, cfg_scale=scale, seed=args.seed, batch_size=args.batch_size,
                                  resume=args.resume, checkpoint_step=args.checkpoint_step))
    del pipe  # free DiT/CLIP/VAE before loading metric networks
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if args.command == "sweep" and args.metrics:
        metrics = [compute_metrics(p, device=args.device, batch_size=args.metric_batch_size) for p in paths]
        from eval.plot import plot_results
        plot_results(metrics, args.output / "figuras")


if __name__ == "__main__":
    main()
