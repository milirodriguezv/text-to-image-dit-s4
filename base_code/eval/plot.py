"""Export comparable evaluation results as CSV and PNG plots."""
import argparse
import csv
from pathlib import Path

from eval.common import read_json


def plot_results(paths, output, axis="cfg"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if axis not in ("cfg", "step"):
        raise ValueError("axis must be cfg or step")
    results = [read_json(path) for path in paths]
    if not results:
        raise ValueError("Provide at least one metrics.json")
    comparable = ["manifest_sha256", "count", "preprocess", "sampler", "steps", "seed", "batch_size",
                  "versions", "code_sha256", "device", "runtime", "null_condition"]
    comparable += ["checkpoint_sha256"] if axis == "cfg" else ["cfg_scale"]
    baseline = results[0]
    for result in results:
        if any(result["config"][k] != baseline["config"][k] for k in comparable):
            raise ValueError("Results use incompatible sample counts, data or sampling configurations")
        if result["fid_implementation"] != baseline["fid_implementation"] or result["metric_versions"] != baseline["metric_versions"]:
            raise ValueError("Results use incompatible metric implementations/versions")
        if axis == "step" and result["config"].get("checkpoint_step") is None:
            raise ValueError("Convergence plots require --checkpoint-step during generation")
    key = "cfg_scale" if axis == "cfg" else "checkpoint_step"
    results.sort(key=lambda r: r["config"][key])
    values = [r["config"][key] for r in results]
    if len(set(values)) != len(values):
        raise ValueError("Duplicate CFG/step values; provide one result per point")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fields = ["checkpoint", "checkpoint_step", "count", "sampler", "steps", "cfg_scale", "fid", "clip_l14", "clip_b32"]
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] if k in r else r["config"].get(k) for k in fields})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].plot(values, [r["fid"] for r in results], "o-")
    axes[0].set_ylabel(baseline["fid_label"] + " (lower is better)")
    for metric, label in (("clip_l14", "CLIP L/14"), ("clip_b32", "CLIP B/32")):
        axes[1].plot(values, [r[metric] for r in results], "o-", label=label)
    for ax in axes:
        ax.set_xlabel("CFG scale" if axis == "cfg" else "Training step")
        ax.grid(alpha=0.25)
    axes[1].set_ylabel("CLIP Score (higher is better)")
    axes[1].legend()
    fig.savefig(output / ("cfg_metrics.png" if axis == "cfg" else "convergence.png"), dpi=180)
    plt.close(fig)
    if axis == "cfg":
        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
        for metric, label in (("clip_l14", "CLIP L/14"), ("clip_b32", "CLIP B/32")):
            ax.plot([r["fid"] for r in results], [r[metric] for r in results], "o-", label=label)
            for r in results:
                ax.annotate(f"s={r['config']['cfg_scale']:g}", (r["fid"], r[metric]), xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel(baseline["fid_label"] + " (lower is better)")
        ax.set_ylabel("CLIP Score (higher is better)")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.savefig(output / "fid_vs_clip.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--axis", choices=["cfg", "step"], default="cfg")
    args = parser.parse_args()
    plot_results(args.metrics, args.output, args.axis)


if __name__ == "__main__":
    main()
