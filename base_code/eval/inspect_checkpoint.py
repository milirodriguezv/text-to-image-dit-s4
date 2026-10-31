"""Inspect a trusted team checkpoint without downloading CLIP/VAE or changing weights."""
import argparse
from pathlib import Path

import torch

from download import find_model
from eval.common import write_json
from models import DiT_models
from sample_t2i import package_versions, sha256


def inspect_checkpoint(path):
    path = Path(path).resolve()
    before = sha256(path)
    state, training = find_model(str(path), return_metadata=True)
    model = DiT_models["DiT-S/4"](input_size=32, text_dim=768)
    model.load_state_dict(state, strict=True)
    nonfinite = [name for name, value in state.items() if not bool(torch.isfinite(value).all())]
    if nonfinite:
        raise ValueError(f"Non-finite weights in checkpoint: {nonfinite}")
    null = model.y_embedder.null_embedding
    if sha256(path) != before:
        raise RuntimeError("Checkpoint changed while reading; use an immutable copy")
    warnings = []
    if not torch.count_nonzero(null):
        warnings.append("The checkpoint uses a zero null condition. Preserve it for evaluation with --allow-zero-null; do not replace it with CLIP empty text after training.")
    return {"schema": 1, "checkpoint": str(path), "checkpoint_sha256": before,
            "size_bytes": path.stat().st_size, "training": training,
            "model": "DiT-S/4", "strict_state_dict_match": True,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "state_dict_entries": len(state), "all_weights_finite": True,
            "null_shape": list(null.shape), "null_norm": float(null.norm()),
            "null_condition": "zeros_from_training_checkpoint" if not torch.count_nonzero(null) else "embedding_from_training_checkpoint",
            "warnings": warnings, "versions": package_versions()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inspect_checkpoint(args.ckpt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(f"Model: {report['model']}; step: {report['training'].get('step')}; weights: {report['training'].get('weights')}")
    print(f"Strict loading passed; {report['parameter_count']:,} parameters; null condition: {report['null_condition']}")
    for message in report["warnings"]:
        print(f"NOTE: {message}")
    print(f"Inspection saved: {args.output}")


if __name__ == "__main__":
    main()
