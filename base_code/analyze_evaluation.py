"""Analyze existing DPM20/CFG4 images: real CLIP baseline and balanced person groups.

Does not generate images, alter the original evaluation or load the DiT checkpoint.
Feature caches stay outside the versioned result directory.
"""
import argparse
import csv
import gc
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image

from eval.common import load_manifest, read_json, write_json
from sample_t2i import CLIP_NAME, package_versions, resolve_device, sha256


GROUPS = ("with_person_annotations", "without_person_annotations")
CLIP_MODELS = (("clip_l14", CLIP_NAME), ("clip_b32", "openai/clip-vit-base-patch32"))


def select_groups(records, annotations, seed):
    ids = [row["image_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate reference IDs")
    annotated_images = {row["id"] for row in annotations["images"]}
    if not set(ids) <= annotated_images:
        raise ValueError("Reference IDs absent from the keypoint image inventory")
    categories = {row["id"]: row["name"] for row in annotations["categories"]}
    if not annotations["annotations"] or any(
        categories.get(row["category_id"]) != "person" for row in annotations["annotations"]
    ):
        raise ValueError("Expected the official person keypoint annotations")
    person_ids = {row["image_id"] for row in annotations["annotations"]}
    populations = {
        GROUPS[0]: sorted(set(ids) & person_ids),
        GROUPS[1]: sorted(set(ids) - person_ids),
    }
    n = min(map(len, populations.values()))
    if n < 2:
        raise ValueError("Both groups need at least two images")
    rng = random.Random(seed)
    selected = {name: sorted(rng.sample(populations[name], n)) for name in GROUPS}
    return {"seed": seed, "n_per_group": n,
            "criterion": "At least one annotation in person_keypoints_val2014.json; no keypoint-count threshold",
            "population_counts": {name: len(rows) for name, rows in populations.items()},
            "selected_ids": selected,
            "population_ids": populations}


def clip_mean(scores, indices=None):
    """TorchMetrics 1.9 clamps the mean raw cosine score, not individual rows."""
    values = np.asarray(scores)
    if indices is not None:
        values = values[indices]
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Expected finite per-image CLIP scores")
    return max(float(np.mean(values, dtype=np.float64)), 0.0)


def indices_for_ids(records, ids):
    positions = {row["image_id"]: i for i, row in enumerate(records)}
    if len(ids) != len(set(ids)) or not set(ids) <= positions.keys():
        raise ValueError("Invalid selected IDs")
    return np.array([positions[image_id] for image_id in ids], dtype=np.int64)


def fid_from_features(real, generated, indices=None):
    from cleanfid.fid import frechet_distance
    real, generated = np.asarray(real), np.asarray(generated)
    if indices is not None:
        real, generated = real[indices], generated[indices]
    if (real.shape != generated.shape or real.ndim != 2 or len(real) < 2
            or not np.isfinite(real).all() or not np.isfinite(generated).all()):
        raise ValueError("FID requires aligned finite feature matrices")
    value = float(frechet_distance(np.mean(real, axis=0), np.cov(real, rowvar=False),
                                   np.mean(generated, axis=0), np.cov(generated, rowvar=False)))
    if not math.isfinite(value):
        raise ValueError("Non-finite FID")
    return value


def save_array(path, array):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)
    write_json(path.with_suffix(".json"), {"sha256": sha256(path), "shape": list(array.shape)})


def load_array(path, shape):
    path = Path(path)
    if not path.exists():
        return None
    metadata = read_json(path.with_suffix(".json"))
    if metadata["sha256"] != sha256(path):
        raise ValueError(f"Corrupted cache: {path.name}")
    array = np.load(path, allow_pickle=False)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"Unexpected cached array: {path.name}")
    return array


def release(device):
    import torch
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def clip_vectors(folders, records, cache, device, batch_size, progress):
    import torch
    from torchmetrics.multimodal import CLIPScore
    # Pinned TorchMetrics implementation: retains its exact image/text encoding,
    # normalization and raw per-pair scores, enabling subgroup means without reruns.
    from torchmetrics.functional.multimodal.clip_score import _clip_score_update
    result = {kind: {} for kind in folders}
    block_size = 512  # divisible by the verified metric batch size (32)
    for label, model_name in CLIP_MODELS:
        metric = None
        for kind, folder in folders.items():
            parts = []
            for start in range(0, len(records), block_size):
                stop = min(start + block_size, len(records))
                path = cache / f"{label}_{kind}_{start:05d}_{stop:05d}.npy"
                values = load_array(path, (stop - start,))
                if values is None:
                    if metric is None:
                        metric = CLIPScore(model_name_or_path=model_name).to(device).eval()
                    batches = []
                    with torch.inference_mode():
                        for i in range(start, stop, batch_size):
                            batch = records[i:min(i + batch_size, stop)]
                            arrays = []
                            for row in batch:
                                with Image.open(folder / row["file"]) as image:
                                    arrays.append(np.array(image, dtype=np.uint8))
                            images = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(device)
                            captions = [row["caption"] for row in batch]
                            scores, count = _clip_score_update(images, captions, metric.model, metric.processor)
                            if count != len(batch) or scores.shape != (len(batch),):
                                raise ValueError("Unexpected CLIP per-image result")
                            # Verify the first batch against the public Metric API.
                            if start == 0 and i == 0:
                                metric.reset()
                                metric.update(images, captions)
                                if not torch.allclose(metric.compute(), scores.mean().clamp(min=0), atol=1e-5, rtol=1e-5):
                                    raise ValueError("Per-image scoring differs from TorchMetrics")
                                metric.reset()
                            batches.append(scores.cpu().numpy())
                    values = np.concatenate(batches)
                    if not np.isfinite(values).all():
                        raise ValueError("Non-finite CLIP scores")
                    save_array(path, values)
                parts.append(values)
                progress(stage=f"{label}_{kind}", done=stop, total=len(records))
            result[kind][label] = np.concatenate(parts)
            print(f"{label} {kind}: {clip_mean(result[kind][label]):.6f}", flush=True)
        del metric
        release(device)
    return result


def inception_vectors(folders, records, cache, device, batch_size, progress):
    import os
    import certifi
    from cleanfid import fid, features
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    extractor = None
    result = {}
    for kind, folder in folders.items():
        parts = []
        for start in range(0, len(records), 512):
            stop = min(start + 512, len(records))
            path = cache / f"inception_{kind}_{start:05d}_{stop:05d}.npy"
            values = load_array(path, (stop - start, 2048))
            if values is None:
                if extractor is None:
                    extractor = features.build_feature_extractor("clean", device=device, use_dataparallel=False)
                files = [str(folder / row["file"]) for row in records[start:stop]]
                values = fid.get_files_features(files, model=extractor, num_workers=0,
                                               batch_size=batch_size, device=device,
                                               mode="clean", verbose=False)
                if values.shape != (stop - start, 2048) or not np.isfinite(values).all():
                    raise ValueError("Unexpected Inception features")
                save_array(path, values)
            parts.append(values)
            progress(stage=f"inception_{kind}", done=stop, total=len(records))
        result[kind] = np.concatenate(parts)
    del extractor
    release(device)
    return result


def write_scores(path, records, groups, scores):
    populations = set(groups["population_ids"][GROUPS[0]])
    selected = set().union(*map(set, groups["selected_ids"].values()))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", "file", "caption", "has_person_annotation", "selected_balanced",
                         "real_clip_l14_raw", "generated_clip_l14_raw", "real_clip_b32_raw", "generated_clip_b32_raw"])
        for i, row in enumerate(records):
            writer.writerow([row["image_id"], row["file"], row["caption"],
                             int(row["image_id"] in populations), int(row["image_id"] in selected),
                             *[float(scores[kind][label][i]) for label, _ in CLIP_MODELS for kind in ("real", "generated")]])


def write_report(output, result):
    output = Path(output)
    fields = ["group", "count", "fid", "real_clip_l14", "generated_clip_l14", "real_clip_b32", "generated_clip_b32"]
    with (output / "metricas_grupos.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["groups"])
    global_scores = result["global"]
    n = result["selection"]["n_per_group"]
    lines = ["# Análisis complementario — DiT-S/4 EMA 150K", "",
             "Se reutilizan las 10.000 imágenes reales y las 10.000 generadas con DPM-Solver++20, CFG 4. No se ha entrenado ni generado de nuevo.", "",
             "## CLIP de las imágenes reales y generadas (10.000 parejas)", "",
             "| Imágenes | CLIP L/14 ↑ | CLIP B/32 ↑ |", "|---|---:|---:|"]
    for kind, label in (("real", "Reales"), ("generated", "Generadas")):
        lines.append(f"| {label} | {global_scores[kind]['clip_l14']:.4f} | {global_scores[kind]['clip_b32']:.4f} |")
    lines += ["", "CLIP real es una referencia del mismo conjunto y captions; no es un techo teórico ni permite convertir el resultado generado en un porcentaje de calidad.", "",
              f"## Grupos equilibrados: {n:,} imágenes por grupo", "",
              "| Grupo por anotaciones | N | FID-N ↓ | CLIP L/14 generadas | CLIP L/14 reales | CLIP B/32 generadas | CLIP B/32 reales |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in result["groups"]:
        label = "Con anotaciones de personas" if row["group"] == GROUPS[0] else "Sin anotaciones de personas"
        lines.append(f"| {label} | {row['count']} | {row['fid']:.4f} | {row['generated_clip_l14']:.4f} | {row['real_clip_l14']:.4f} | {row['generated_clip_b32']:.4f} | {row['real_clip_b32']:.4f} |")
    lines += ["", "## Protocolo y alcance", "",
              f"- Dentro de las 10K referencias hay {result['selection']['population_counts'][GROUPS[0]]} IDs con anotaciones y {result['selection']['population_counts'][GROUPS[1]]} sin ellas. La selección se realiza sobre IDs ordenados, con `random.Random({result['selection']['seed']})`, y se guarda en `ids_grupos.json`.",
              "- La etiqueta se obtiene de al menos una anotación en `person_keypoints_val2014.json`, sin exigir keypoints visibles. Describe la foto original: no garantiza una persona visible tras el recorte central ni que la caption hable de personas. Sin anotaciones no es una inspección visual de ausencia de personas.",
              "- Se mantienen los PNG originales y el recorte central de la evaluación general. No se reproduce aquí el recorte de personas del entrenamiento.",
              "- FID-N usa solo las N reales y las N generadas de los mismos IDs de cada grupo, con clean-fid/clean/Inception-v3. No debe compararse directamente con FID-10K; igualar N no elimina la incertidumbre ni las diferencias de composición entre grupos.",
              "- Los resultados son descriptivos para una selección y una semilla; no prueban una causa de los fallos ni significación estadística.",
              "- CLIP usa los mismos modelos y procesamiento que la evaluación original. Se guardan scores por imagen antes del clamp; cada media se calcula como `max(mean(100*cos), 0)`, igual que TorchMetrics 1.9. Los valores de L/14 y B/32 se interpretan por separado.",
              "- Se comprobó la reproducción de las métricas globales originales antes de aceptar este análisis. Las cachés de features no se versionan.", "",
              "## Archivos", "",
              "`resultados.json` conserva métricas, controles y procedencia; `metricas_grupos.csv` permite incorporar la tabla al informe; `scores_por_imagen.csv` permite auditar las medias; `ids_grupos.json` fija las selecciones.", ""]
    (output / "INFORME.md").write_text("\n".join(lines), encoding="utf-8")


def analyze(run_path, annotations_path, output, cache, *, device="mps", batch_size=32, seed=0):
    import torch
    from eval.run import validate_run
    start_time = time.monotonic()
    run_path, annotations_path = Path(run_path).resolve(), Path(annotations_path).resolve()
    output, cache = Path(output).resolve(), Path(cache).resolve()
    if cache.is_relative_to(output):
        raise ValueError("Feature caches must stay outside the published result folder")
    if batch_size < 1 or 512 % batch_size:
        raise ValueError("batch_size must be a positive divisor of 512")
    if annotations_path.name != "person_keypoints_val2014.json":
        raise ValueError("Use official person_keypoints_val2014.json")
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "phase": "validate_inputs"}

    def progress(**changes):
        status.update(changes)
        status["elapsed_seconds"] = round(time.monotonic() - start_time, 2)
        write_json(output / "status.json", status)
        print(json.dumps(status), flush=True)

    progress()
    try:
        run, records = validate_run(run_path)
        cfg = run["config"]
        if (cfg["sampler"], cfg["steps"], cfg["cfg_scale"], cfg["count"], cfg["weights"]) != ("dpm", 20, 4, 10000, "ema"):
            raise ValueError("Expected completed DPM20/CFG4/EMA with 10K records")
        load_manifest(cfg["manifest"], verify=True)
        original = read_json(run_path.parent / "metrics.json")
        if original["run_sha256"] != sha256(run_path) or original["config"] != cfg:
            raise ValueError("Original metrics do not match the frozen run")
        device = resolve_device(device)
        versions = package_versions()
        if versions != original["metric_versions"]:
            raise ValueError("Use the same metric versions as the original evaluation")
        if str(device) != original["clip_device"] or str(device) != original["metric_device"]:
            raise ValueError("Use the original metric device for the reproduction control")
        groups = select_groups(records, read_json(annotations_path), seed)
        context = {"schema": 1, "run_sha256": sha256(run_path), "manifest_sha256": cfg["manifest_sha256"],
                   "original_metrics_sha256": sha256(run_path.parent / "metrics.json"),
                   "person_annotations_sha256": sha256(annotations_path),
                   "person_annotations_file": annotations_path.name,
                   "checkpoint_sha256": cfg["checkpoint_sha256"], "checkpoint_step": cfg["checkpoint_step"],
                   "sampler": "dpm", "steps": 20, "cfg_scale": 4, "count": 10000,
                   "null_condition": cfg["null_condition"], "preprocess": cfg["preprocess"],
                   "device": str(device), "batch_size": batch_size, "seed": seed,
                   "versions": versions, "analysis_code_sha256": sha256(__file__),
                   "scoring_rule": "TorchMetrics 1.9 raw pair cosine scores, clamp after averaging"}
        for path in (output / "config.json", cache / "config.json"):
            if path.exists() and read_json(path) != context:
                raise ValueError("Analysis/cache inputs changed; use a new directory")
            write_json(path, context)
        write_json(output / "ids_grupos.json", groups)
        folders = {"real": Path(cfg["manifest"]).parent / "real", "generated": run_path.parent / "images"}
        progress(phase="clip")
        scores = clip_vectors(folders, records, cache, device, batch_size, progress)
        global_scores = {kind: {label: clip_mean(values[label]) for label, _ in CLIP_MODELS}
                         for kind, values in scores.items()}
        control = {}
        for label, _ in CLIP_MODELS:
            delta = abs(global_scores["generated"][label] - original[label])
            control[label] = {"original": original[label], "recomputed": global_scores["generated"][label],
                              "absolute_difference": delta, "tolerance": 0.001}
            if delta > 0.001:
                raise ValueError(f"Original CLIP reproduction failed: {label}")
        write_scores(output / "scores_por_imagen.csv", records, groups, scores)
        write_json(output / "clip_global.json", {"count": len(records), "scores": global_scores, "control": control})
        progress(phase="inception")
        features = inception_vectors(folders, records, cache, device, batch_size, progress)
        progress(phase="fid", stage="fid_global_control", done=0, total=3)
        fid_global = fid_from_features(features["real"], features["generated"])
        delta = abs(fid_global - original["fid"])
        control["fid_10k"] = {"original": original["fid"], "recomputed": fid_global,
                              "absolute_difference": delta, "tolerance": 0.01}
        if delta > 0.01:
            raise ValueError("Original FID reproduction failed")
        rows = []
        for i, group in enumerate(GROUPS):
            progress(stage=f"fid_{group}", done=i + 1, total=3)
            indices = indices_for_ids(records, groups["selected_ids"][group])
            rows.append({"group": group, "count": len(indices),
                         "fid": fid_from_features(features["real"], features["generated"], indices),
                         **{f"{kind}_{label}": clip_mean(values[label], indices)
                            for kind, values in scores.items() for label, _ in CLIP_MODELS}})
        if (sha256(run_path) != context["run_sha256"]
                or sha256(cfg["manifest"]) != context["manifest_sha256"]
                or sha256(annotations_path) != context["person_annotations_sha256"]):
            raise ValueError("Inputs changed during analysis")
        selection = {key: groups[key] for key in ("seed", "n_per_group", "criterion", "population_counts")}
        selection.update(ids_file="ids_grupos.json", ids_file_sha256=sha256(output / "ids_grupos.json"))
        result = {"schema": 1, "status": "complete", "config": context, "selection": selection,
                  "global": global_scores, "groups": rows, "reproduction_control": control,
                  "elapsed_seconds": time.monotonic() - start_time}
        write_json(output / "resultados.json", result)
        write_report(output, result)
        inventory = {p.name: sha256(p) for p in sorted(output.iterdir())
                     if p.is_file() and p.name not in {"status.json", "archivos_sha256.json"}}
        write_json(output / "archivos_sha256.json", inventory)
        progress(status="complete", phase="complete", stage="complete", done=10000, total=10000)
        return result
    except Exception as error:
        progress(status="failed", error=f"{type(error).__name__}: {error}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--person-annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    analyze(args.run, args.person_annotations, args.output, args.cache,
            device=args.device, batch_size=args.batch_size, seed=args.seed)


if __name__ == "__main__":
    main()
