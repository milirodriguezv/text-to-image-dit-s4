"""Run Persona 4's agreed evaluation: FID-2K, six FID-10K CFGs and DDPM-250."""
import argparse
from datetime import datetime
import gc
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

from eval.common import read_json, write_json, load_manifest
from eval.plot import plot_results
from eval.run import CFG_SCALES, compute_metrics, generate_run, validate_run
from sample_t2i import TextToImagePipeline, sha256


def execute(ckpt, manifest, output, *, generation_device="cpu", metric_device="cpu",
            batch_size=8, metric_batch_size=32, allow_zero_null=False, clip_device=None,
            intermediate_metrics=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data, records = load_manifest(manifest)
    if len(records) != 10000:
        raise ValueError("The agreed final protocol requires a frozen 10,000-image manifest")
    state = {"status": "running", "checkpoint_sha256": sha256(ckpt),
             "manifest_sha256": sha256(manifest), "preprocess": data["preprocess"],
             "generation_device": generation_device, "metric_device": metric_device,
             "clip_device": clip_device or metric_device,
             "finished_evaluations": [], "started": datetime.now(ZoneInfo("Europe/Madrid")).isoformat()}

    def update(**changes):
        state.update(changes)
        state["updated"] = datetime.now(ZoneInfo("Europe/Madrid")).isoformat()
        write_json(output / "status.json", state)

    def evaluate(name, sampler, steps, cfg, count):
        folder = output / name
        update(phase="generation", evaluation=name, images_done=0, images_target=count)
        pipe = TextToImagePipeline.from_checkpoint(ckpt, generation_device,
                                                   allow_zero_null=allow_zero_null)
        state["training_step"] = pipe.training.get("step")
        try:
            generate_run(pipe, manifest, folder, count=count, sampler=sampler, steps=steps,
                         cfg_scale=cfg, seed=0, batch_size=batch_size,
                         resume=(folder / "run.json").is_file(),
                         progress_callback=lambda done, target: update(images_done=done, images_target=target))
        finally:
            del pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        update(phase="metrics", evaluation=name, images_done=count)
        metrics_path = folder / "metrics.json"
        # Recompute metrics only when generation/config changed or no result exists.
        if not metrics_path.is_file() or read_json(metrics_path).get("run_sha256") != sha256(folder / "run.json"):
            compute_metrics(folder / "run.json", device=metric_device, batch_size=metric_batch_size, clip_device=clip_device)
        result = read_json(metrics_path)
        state["finished_evaluations"].append({"name": name, "count": count, "sampler": sampler,
                                              "steps": steps, "cfg_scale": cfg,
                                              "fid": result["fid"], "clip_l14": result["clip_l14"],
                                              "clip_b32": result["clip_b32"]})
        update()
        return metrics_path

    try:
        if intermediate_metrics is None:
            evaluate("intermediate_cfg4_2k", "dpm", 20, 4, 2000)
        else:
            previous = read_json(intermediate_metrics)
            run, _ = validate_run(previous["run"])
            config = run["config"]
            if (previous["run_sha256"] != sha256(previous["run"])
                    or config["checkpoint_sha256"] != state["checkpoint_sha256"]
                    or config["manifest_sha256"] != state["manifest_sha256"]
                    or (config["count"], config["sampler"], config["steps"], config["cfg_scale"]) != (2000, "dpm", 20, 4)):
                raise ValueError("Existing FID-2K result does not match this checkpoint and protocol")
            state["finished_evaluations"].append({"name": "intermediate_cfg4_2k", "count": 2000,
                                                  "sampler": "dpm", "steps": 20, "cfg_scale": 4,
                                                  **{key: previous[key] for key in ("fid", "clip_l14", "clip_b32")}})
            state["training_step"] = config["checkpoint_step"]
        paths = [evaluate(f"final_dpm_cfg_{cfg:g}", "dpm", 20, cfg, 10000) for cfg in CFG_SCALES]
        plot_results(paths, output / "figuras")
        best = min((read_json(path) for path in paths), key=lambda result: result["fid"])
        cfg = best["config"]["cfg_scale"]
        update(ddpm_cfg_selection="lowest FID-10K in the completed DPM sweep", ddpm_cfg=cfg)
        evaluate(f"final_ddpm_cfg_{cfg:g}", "ddpm", 250, cfg, 10000)
        write_json(output / "results.json", state["finished_evaluations"])
        update(status="complete", phase="complete")
    except Exception as error:
        update(status="failed", error=f"{type(error).__name__}: {error}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generation-device", default="cpu")
    parser.add_argument("--metric-device", default="cpu")
    parser.add_argument("--clip-device", help="CLIP scoring device; defaults to metric-device")
    parser.add_argument("--intermediate-metrics", type=Path, help="Reuse a verified completed FID-2K result")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--metric-batch-size", type=int, default=32)
    parser.add_argument("--allow-zero-null", action="store_true")
    args = parser.parse_args()
    execute(args.ckpt, args.manifest, args.output, generation_device=args.generation_device,
            metric_device=args.metric_device, batch_size=args.batch_size,
            metric_batch_size=args.metric_batch_size, allow_zero_null=args.allow_zero_null,
            clip_device=args.clip_device, intermediate_metrics=args.intermediate_metrics)


if __name__ == "__main__":
    main()
