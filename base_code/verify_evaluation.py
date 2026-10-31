"""Verify saved evaluation artifacts without generating or rescoring images."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def check_png(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Image hash differs: {path}")
    with path.open("rb") as stream:
        header = stream.read(29)
    if (header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR"
            or struct.unpack(">II", header[16:24]) != (256, 256)
            or header[24:26] != b"\x08\x02"):
        raise ValueError(f"Expected an RGB 8-bit 256x256 PNG: {path}")


def verify(folder, checkpoint=None):
    folder = Path(folder).resolve()
    status = read(folder / "status.json")
    if status["status"] != "complete":
        raise ValueError("Evaluation is not complete")
    expected = {f"final_dpm_cfg_{s:g}": ("dpm", 20, s, 10000) for s in (1, 1.5, 2, 3, 4, 5)}
    expected.update({"intermediate_cfg4_2k": ("dpm", 20, 4, 2000),
                     "final_ddpm_cfg_4": ("ddpm", 250, 4, 10000)})
    finished = {row["name"]: row for row in status["finished_evaluations"]}
    if set(finished) != set(expected) or len(status["finished_evaluations"]) != len(expected):
        raise ValueError("Missing, extra or duplicate evaluations")
    first = read(folder / "final_dpm_cfg_4/run.json")
    manifest_path = Path(first["config"]["manifest"])
    manifest = read(manifest_path)
    if digest(manifest_path) != status["manifest_sha256"] or manifest["count"] != 10000:
        raise ValueError("Reference manifest changed")
    references = manifest["records"]
    if len(references) != 10000 or len({r["image_id"] for r in references}) != 10000:
        raise ValueError("Reference count/IDs invalid")
    for row in references:
        check_png(manifest_path.parent / "real" / row["file"], row["sha256"])
    print("References verified: 10000/10000", flush=True)
    checkpoint_path = Path(checkpoint) if checkpoint else Path(first["config"]["checkpoint"])
    if digest(checkpoint_path) != status["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed")
    baseline = first["config"]
    invariant = ("manifest_sha256", "checkpoint_sha256", "checkpoint_step", "model", "weights",
                 "preprocess", "seed", "batch_size", "versions", "code_sha256", "device", "runtime",
                 "null_condition", "clip", "vae")
    counts = {}
    for name, spec in expected.items():
        run_path = folder / name / "run.json"
        run, metrics = read(run_path), read(folder / name / "metrics.json")
        cfg = run["config"]
        if run["status"] != "complete" or metrics["run_sha256"] != digest(run_path):
            raise ValueError(f"Incomplete or changed generation metadata: {name}")
        comparable = invariant if spec[3] == 10000 else tuple(k for k in invariant if k != "code_sha256")
        if metrics["config"] != cfg or any(cfg[k] != baseline[k] for k in comparable):
            raise ValueError(f"Configuration differs: {name}")
        if (cfg["sampler"], cfg["steps"], cfg["cfg_scale"], cfg["count"]) != spec:
            raise ValueError(f"Wrong sampling protocol: {name}")
        if cfg["checkpoint_step"] != 150000 or cfg["weights"] != "ema":
            raise ValueError("Expected the available 150K EMA checkpoint")
        selected, generated = references[:cfg["count"]], run["records"]
        if len(generated) != cfg["count"]:
            raise ValueError(f"Wrong generated count: {name}")
        image_dir = folder / name / "images"
        if {p.name for p in image_dir.iterdir()} != {r["file"] for r in selected}:
            raise ValueError(f"Image set differs: {name}")
        for i, (reference, actual) in enumerate(zip(selected, generated)):
            if any(reference[k] != actual[k] for k in ("file", "image_id", "caption")):
                raise ValueError(f"Image/caption pairing differs: {name}")
            if actual["batch_seed"] != cfg["seed"] + (i // cfg["batch_size"]) * cfg["batch_size"]:
                raise ValueError(f"Batch seed differs: {name}")
            check_png(image_dir / actual["file"], actual["sha256"])
        for key in ("fid", "clip_l14", "clip_b32"):
            if not math.isfinite(metrics[key]) or not math.isclose(metrics[key], finished[name][key], abs_tol=1e-10):
                raise ValueError(f"Metric/status mismatch: {name}/{key}")
        if metrics["fid_implementation"] != "clean-fid/clean/inception_v3":
            raise ValueError("Unexpected FID implementation")
        counts[name] = len(generated)
        print(f"Verified {name}: {len(generated)} PNGs + metrics", flush=True)
    with (folder / "results.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(expected) or {r["evaluation"] for r in rows} != set(expected):
        raise ValueError("CSV rows differ")
    for row in rows:
        for key in ("fid", "clip_l14", "clip_b32"):
            if not math.isclose(float(row[key]), finished[row["evaluation"]][key], abs_tol=1e-10):
                raise ValueError("CSV metrics differ")
    for name in ("cfg_metrics.png", "fid_vs_clip.png", "results.csv"):
        if not (folder / "figuras" / name).is_file():
            raise ValueError(f"Missing figure/CSV: {name}")
    return {"status": "PASS", "reference_images_verified": 10000,
            "generated_images_verified": sum(counts.values()), "evaluations": counts,
            "checkpoint_sha256": status["checkpoint_sha256"],
            "manifest_sha256": status["manifest_sha256"],
            "code_sha256": baseline["code_sha256"],
            "intermediate_code_sha256": read(folder / "intermediate_cfg4_2k/run.json")["config"]["code_sha256"],
            "scope": "Integrity, PNG headers, pairing, seeds, protocol and saved metrics/CSV. Metrics are not recomputed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    start = time.perf_counter()
    result = verify(args.folder)
    result["seconds"] = time.perf_counter() - start
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
