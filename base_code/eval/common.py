"""Manifest I/O and integrity checks shared by evaluation commands."""
import json
from pathlib import Path

from PIL import Image

from sample_t2i import sha256


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked_image(path, expected_hash=None):
    path = Path(path)
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (256, 256) or image.format != "PNG":
            raise ValueError(f"Expected RGB 256x256 PNG: {path}")
    if expected_hash is not None and sha256(path) != expected_hash:
        raise ValueError(f"Image has changed since it was recorded: {path}")


def load_manifest(path, count=None, verify=True):
    path = Path(path).resolve()
    data = read_json(path)
    records = data.get("records", [])
    if data.get("schema") != 1 or data.get("split") != "val2014" or not records:
        raise ValueError("Expected a schema=1 val2014 manifest from eval.prepare")
    if data.get("count") != len(records):
        raise ValueError("Manifest count does not match records")
    ids = [r["image_id"] for r in records]
    names = [r["file"] for r in records]
    if len(set(ids)) != len(ids) or len(set(names)) != len(names):
        raise ValueError("Manifest contains duplicate images")
    for row in records:
        if not isinstance(row["caption"], str) or not row["caption"].strip():
            raise ValueError("Empty caption in manifest")
        if Path(row["file"]).name != row["file"] or not row["file"].endswith(".png"):
            raise ValueError("Manifest image names must be plain PNG filenames")
    if count is not None and not 2 <= count <= len(records):
        raise ValueError(f"count must be between 2 and {len(records)}")
    selected = records if count is None else records[:count]
    if verify:
        for row in selected:
            checked_image(path.parent / "real" / row["file"], row["sha256"])
    return data, selected
