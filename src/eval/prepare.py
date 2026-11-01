"""Freeze a deterministic val2014 subset, captions and reference image preprocessing."""
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

from eval.common import read_json, sha256, write_json


def center_crop(image, size=256):
    """ADM preprocessing, matching the original train.py center_crop_arr."""
    image = image.convert("RGB")
    while min(image.size) >= 2 * size:
        image = image.resize(tuple(x // 2 for x in image.size), resample=Image.Resampling.BOX)
    scale = size / min(image.size)
    image = image.resize(tuple(round(x * scale) for x in image.size), resample=Image.Resampling.BICUBIC)
    left, top = (image.width - size) // 2, (image.height - size) // 2
    return image.crop((left, top, left + size, top + size))


def prepare(captions_path, images_dir, output, *, count=10000, seed=0,
            preprocess, person_annotations=None):
    captions_path, images_dir, output = map(Path, (captions_path, images_dir, output))
    # Require the COCO split explicitly to prevent accidentally preparing training references.
    if captions_path.name != "captions_val2014.json":
        raise ValueError("Use the original captions_val2014.json, not training/2017 annotations")
    if output.exists():
        raise FileExistsError(f"Use a new directory to preserve the evaluation subset: {output}")
    if preprocess not in ("center-crop", "person-crop"):
        raise ValueError("Choose center-crop or person-crop explicitly")
    coco = read_json(captions_path)
    first_caption = {}
    for item in coco["annotations"]:
        first_caption.setdefault(item["image_id"], item["caption"])
    boxes = {}
    if preprocess == "person-crop":
        if person_annotations is None or Path(person_annotations).name != "person_keypoints_val2014.json":
            raise ValueError("person-crop requires person_keypoints_val2014.json")
        for annotation in read_json(person_annotations)["annotations"]:
            boxes.setdefault(annotation["image_id"], []).append(annotation["bbox"])
    eligible = sorted((item for item in coco["images"] if item["id"] in first_caption
                       and (preprocess != "person-crop" or item["id"] in boxes)), key=lambda x: x["id"])
    if not 2 <= count <= len(eligible):
        raise ValueError(f"Requested {count} references, but only {len(eligible)} eligible images")
    random.Random(seed).shuffle(eligible)
    selected = eligible[:count]
    # Fail for missing source files rather than silently changing the chosen subset.
    for item in selected:
        if Path(item["file_name"]).name != item["file_name"]:
            raise ValueError("Invalid COCO image filename")
        if not (images_dir / item["file_name"]).is_file():
            raise FileNotFoundError(images_dir / item["file_name"])
    (output / "real").mkdir(parents=True)
    records = []
    for item in selected:
        source = images_dir / item["file_name"]
        with Image.open(source) as image:
            if preprocess == "center-crop":
                processed = center_crop(image)
            else:
                from extract_features import preprocess_image
                tensor = preprocess_image(image, boxes[item["id"]])
                array = ((tensor.permute(1, 2, 0).numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
                processed = Image.fromarray(array)
        name = f"{item['id']:012d}.png"
        destination = output / "real" / name
        processed.save(destination)
        records.append({"image_id": item["id"], "caption": first_caption[item["id"]],
                        "source": str(source.resolve()), "file": name, "sha256": sha256(destination)})
    manifest = {"schema": 1, "split": "val2014", "count": count, "seed": seed,
                "caption_selection": "first in COCO annotation order", "preprocess": preprocess,
                "captions_sha256": sha256(captions_path),
                "person_annotations_sha256": sha256(person_annotations) if preprocess == "person-crop" else None,
                "records": records}
    write_json(output / "manifest.json", manifest)
    return output / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preprocess", choices=["center-crop", "person-crop"], required=True,
                        help="Choose the preprocessing agreed with the training team")
    parser.add_argument("--person-annotations", type=Path)
    args = parser.parse_args()
    path = prepare(args.captions, args.images_dir, args.output, count=args.count, seed=args.seed,
                   preprocess=args.preprocess, person_annotations=args.person_annotations)
    print(f"Frozen evaluation manifest: {path}")


if __name__ == "__main__":
    main()
