"""CLIP-only diagnostic for fixed prompt samples. This is not COCO FID-10K evaluation."""
import argparse
from pathlib import Path

from eval.common import checked_image, read_json, write_json
from eval.run import compute_clip_scores
from sample_t2i import package_versions, sha256


def score_samples(samples_path, output, *, device="auto", batch_size=8):
    samples_path = Path(samples_path).resolve()
    metadata = read_json(samples_path)
    records = []
    for row in metadata["samples"]:
        if Path(row["image"]).name != row["image"]:
            raise ValueError("Sample filenames must be plain relative filenames")
        path = samples_path.parent / row["image"]
        checked_image(path)
        records.append({"file": row["image"], "caption": row["prompt"], "sha256": sha256(path)})
    scores = compute_clip_scores(samples_path.parent, records, device=device, batch_size=batch_size)
    report = {"schema": 1, "diagnostic_only": True,
              "scope": "fixed prompts, not a representative COCO evaluation; no FID",
              "count": len(records), "samples_sha256": sha256(samples_path),
              "checkpoint_sha256": metadata["checkpoint_sha256"], "training": metadata.get("training"),
              "sampler": metadata["sampler"], "steps": metadata["steps"], "cfg_scale": metadata["cfg_scale"],
              "null_condition": metadata.get("null_condition"), "records": records,
              "versions": package_versions(), **scores}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    result = score_samples(args.samples, args.output, device=args.device, batch_size=args.batch_size)
    print(f"Diagnostic only ({result['count']} fixed prompts): L/14={result['clip_l14']:.4f}, B/32={result['clip_b32']:.4f}")


if __name__ == "__main__":
    main()
