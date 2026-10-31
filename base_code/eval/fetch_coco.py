"""Fetch the deterministic val2014 evaluation subset from the official COCO S3 bucket."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import random
import threading
import zipfile

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from eval.common import read_json, write_json
from sample_t2i import sha256

BUCKET = "https://s3.amazonaws.com/images.cocodataset.org"
ANNOTATIONS_URL = f"{BUCKET}/annotations/annotations_trainval2014.zip"


def select_images(coco, count, seed):
    caption_ids = {row["image_id"] for row in coco["annotations"]}
    eligible = sorted((row for row in coco["images"] if row["id"] in caption_ids), key=lambda row: row["id"])
    if not 2 <= count <= len(eligible):
        raise ValueError(f"count must be between 2 and {len(eligible)}")
    random.Random(seed).shuffle(eligible)
    return eligible[:count]


def download_annotations(root):
    root = Path(root)
    folder = root / "annotations"
    folder.mkdir(parents=True, exist_ok=True)
    path = root / "annotations_trainval2014.zip"
    if not path.exists():
        temporary = path.with_suffix(".part")
        with requests.get(ANNOTATIONS_URL, stream=True, timeout=60) as response:
            response.raise_for_status()
            with temporary.open("wb") as stream:
                for chunk in response.iter_content(4 * 1024 * 1024):
                    stream.write(chunk)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("COCO annotation archive failed its CRC check")
        temporary.replace(path)
    with zipfile.ZipFile(path) as archive:
        for name in ("captions_val2014.json", "captions_train2014.json"):
            destination = folder / name
            if not destination.exists():
                temporary = destination.with_suffix(".part")
                temporary.write_bytes(archive.read(f"annotations/{name}"))
                temporary.replace(destination)
    return folder / "captions_val2014.json", folder / "captions_train2014.json", path


def fetch(root, *, count=10000, seed=0, workers=8):
    if not 1 <= workers <= 16:
        raise ValueError("Use between 1 and 16 download workers")
    root = Path(root)
    val_path, train_path, archive_path = download_annotations(root)
    coco = read_json(val_path)
    selected = select_images(coco, count, seed)
    train_ids = {row["id"] for row in read_json(train_path)["images"]}
    overlap = train_ids & {row["id"] for row in selected}
    if overlap:
        raise ValueError(f"Evaluation IDs overlap the official train2014 split: {overlap}")
    images_dir = root / "val2014"
    images_dir.mkdir(exist_ok=True)
    local = threading.local()

    def get_image(row):
        name = row["file_name"]
        if Path(name).name != name or not name.startswith("COCO_val2014_"):
            raise ValueError("Invalid val2014 filename")
        path = images_dir / name
        if path.exists():
            with Image.open(path) as image:
                image.verify()
            return
        if not hasattr(local, "session"):
            local.session = requests.Session()
            retries = Retry(total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
            local.session.mount("https://", HTTPAdapter(max_retries=retries))
        temporary = path.with_suffix(".part")
        with local.session.get(f"{BUCKET}/val2014/{name}", stream=True, timeout=60) as response:
            response.raise_for_status()
            with temporary.open("wb") as stream:
                for chunk in response.iter_content(256 * 1024):
                    stream.write(chunk)
        with Image.open(temporary) as image:
            image.verify()
        temporary.replace(path)

    print(f"Downloading {len(selected)} fixed val2014 images (seed {seed})", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(get_image, row) for row in selected]
        for n, future in enumerate(as_completed(futures), 1):
            future.result()
            if n % 250 == 0 or n == count:
                print(f"COCO {n}/{count}", flush=True)
    metadata = {"schema": 1, "source": BUCKET, "annotations_url": ANNOTATIONS_URL,
                "annotations_zip_sha256": sha256(archive_path), "captions_sha256": sha256(val_path),
                "split": "val2014", "count": count, "seed": seed,
                "official_train2014_overlap": 0, "image_ids": [row["id"] for row in selected]}
    write_json(root / "evaluation_download.json", metadata)
    return val_path, images_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    paths = fetch(args.root, count=args.count, seed=args.seed, workers=args.workers)
    print("Ready:", *paths)


if __name__ == "__main__":
    main()
