"""Checkpoint provenance, portable notebook and diagnostic scoring checks."""
import ast
import base64
import hashlib
import io
import json
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from eval.build_colab import build_notebook
from eval.common import read_json
from eval.fetch_checkpoint import DownloadForm, download_checkpoint
from eval.inspect_checkpoint import inspect_checkpoint


def test_inspection_reports_actual_checkpoint_and_null(model, tmp_path):
    path = tmp_path / "checkpoint.pt"
    with torch.no_grad():
        model.y_embedder.null_embedding.zero_()
    torch.save({"ema": model.state_dict(), "step": 150000,
                "args": Namespace(model="DiT-S/4", global_batch_size=128)}, path)
    report = inspect_checkpoint(path)
    assert report["training"]["step"] == 150000
    assert report["training"]["args"]["global_batch_size"] == 128
    assert report["strict_state_dict_match"] and report["all_weights_finite"]
    assert report["null_condition"] == "zeros_from_training_checkpoint"
    assert report["warnings"] and report["parameter_count"] == 33003776


def test_drive_download_form_and_checkpoint_hash(tmp_path, monkeypatch):
    import eval.fetch_checkpoint as fetch
    body = b"test checkpoint bytes"
    expected = hashlib.sha256(body).hexdigest()
    html = '<form id="download-form" action="https://drive.usercontent.google.com/download"><input name="confirm" value="t"></form>'
    parser = DownloadForm()
    parser.feed(html)
    assert parser.params == {"confirm": "t"}

    class Response:
        def __init__(self, is_html):
            self.headers = {"Content-Type": "text/html" if is_html else "application/octet-stream"}
            self.text = html

        def raise_for_status(self):
            pass

        def close(self):
            pass

        def iter_content(self, size):
            return iter([body])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, **kwargs):
            return Response(url.startswith("https://drive.google.com/uc"))

    monkeypatch.setattr(fetch.requests, "Session", Session)
    path = tmp_path / "last.pt"
    assert download_checkpoint(path, expected_sha256=expected).read_bytes() == body
    assert download_checkpoint(path, expected_sha256=expected) == path
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="not overwritten"):
        download_checkpoint(path, expected_sha256=expected)
    path.unlink()
    with pytest.raises(ValueError, match="does not match"):
        download_checkpoint(path, expected_sha256="0" * 64)
    assert not path.exists() and not path.with_suffix(".part").exists()


def test_portable_colab_contains_current_code_and_valid_cells(tmp_path):
    path = build_notebook(tmp_path / "evaluation.ipynb")
    notebook = json.loads(path.read_text())
    bundle = None
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "b64decode":
                bundle = base64.b64decode(ast.literal_eval(node.args[0]))
    assert bundle is not None
    assert hashlib.sha256(bundle).hexdigest() == notebook["metadata"]["source_bundle_sha256"]
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        assert archive.read("sample_t2i.py") == (Path(__file__).parents[1] / "sample_t2i.py").read_bytes()
        assert archive.read("eval/run.py") == (Path(__file__).parents[1] / "eval/run.py").read_bytes()
        assert not any(name.endswith((".pt", ".ipynb")) or "results/" in name for name in archive.namelist())
        assert b"transformers==4.57.6" in archive.read("requirements-eval.txt")
