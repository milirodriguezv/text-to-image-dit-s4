"""Offline checks for Persona 4; no COCO, pretrained downloads or training needed."""
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from eval.common import read_json, write_json
from eval.prepare import prepare
from eval.run import compute_metrics, generate_run, validate_run
from eval.plot import plot_results
from sample_t2i import (TextToImagePipeline, decode_latents, resolve_device,
                        sample_latents, sha256)


class ToyDenoiser(nn.Module):
    """Nonconstant conditioning plus nonzero variance to exercise both samplers."""
    def forward(self, x, t, y):
        eps = x * 0.01 + y[:, :4, None, None] * 0.001
        return torch.cat([eps, torch.zeros_like(eps)], dim=1)

    def forward_with_cfg(self, x, t, y, cfg_scale):
        first = x.chunk(2)[0]
        output = self(torch.cat([first, first]), t, y)
        cond, uncond = output[:, :4].chunk(2)
        eps = uncond + cfg_scale * (cond - uncond)
        return torch.cat([torch.cat([eps, eps]), output[:, 4:]], dim=1)


@pytest.mark.parametrize("sampler,steps", [("dpm", 20), ("ddpm", 5), ("ddpm", 250)])
def test_samplers_reproducible_and_conditioned(sampler, steps):
    model = ToyDenoiser()
    e, null = torch.ones(2, 768), torch.zeros(1, 768)
    state = torch.random.get_rng_state().clone()
    a = sample_latents(model, e, null, sampler=sampler, steps=steps, cfg_scale=1, seed=13)
    assert torch.equal(state, torch.random.get_rng_state())
    b = sample_latents(model, e, null, sampler=sampler, steps=steps, cfg_scale=1, seed=13)
    guided = sample_latents(model, e, null, sampler=sampler, steps=steps, cfg_scale=4, seed=13)
    other = sample_latents(model, e, null, sampler=sampler, steps=steps, cfg_scale=1, seed=14)
    assert a.shape == (2, 4, 32, 32) and torch.isfinite(a).all()
    assert torch.equal(a, b)
    assert not torch.allclose(a, guided)
    assert not torch.equal(a, other)


def test_dpm_guides_all_channels_but_ignores_variance():
    class Probe(ToyDenoiser):
        def __init__(self, variance):
            super().__init__()
            self.variance = variance
            self.times = []

        def forward(self, x, t, y):
            self.times.extend(t.tolist())
            result = super().forward(x, t, y)
            result[:, 4:] = self.variance
            return result

    e, null = torch.ones(1, 768), torch.zeros(1, 768)
    model = Probe(0)
    a = sample_latents(model, e, null, cfg_scale=1)
    b = sample_latents(Probe(100), e, null, cfg_scale=1)
    c = sample_latents(Probe(0), e, null, cfg_scale=4)
    assert torch.equal(a, b)
    assert all(not torch.allclose(a[:, channel], c[:, channel]) for channel in range(4))
    assert max(model.times) == 999 and min(model.times) >= 0


def test_ddpm_uses_predicted_variance():
    class WithVariance(ToyDenoiser):
        def forward(self, x, t, y):
            result = super().forward(x, t, y)
            result[:, 4:] = 0.5
            return result
    e, null = torch.ones(1, 768), torch.zeros(1, 768)
    a = sample_latents(ToyDenoiser(), e, null, sampler="ddpm", steps=5)
    b = sample_latents(WithVariance(), e, null, sampler="ddpm", steps=5)
    assert not torch.allclose(a, b)


def test_decode_undoes_scale_and_converts_range():
    class VAE:
        def decode(self, z):
            assert torch.allclose(z, torch.ones_like(z))
            return SimpleNamespace(sample=torch.tensor([[[[-2., 0., 2.]]]]))
    pixels = decode_latents(VAE(), torch.full((1, 4, 32, 32), 0.18215))
    assert pixels.tolist() == [[[[0., 0.5, 1.]]]]


@pytest.mark.parametrize("kwargs", [{"steps": 1}, {"cfg_scale": float("nan")}, {"sampler": "invalid"}])
def test_invalid_sampler_settings(kwargs):
    with pytest.raises(ValueError):
        sample_latents(ToyDenoiser(), torch.ones(1, 768), torch.zeros(1, 768), **kwargs)


def test_cpu_and_mps_device_availability():
    assert resolve_device("cpu") == torch.device("cpu")
    if torch.backends.mps.is_available():
        assert resolve_device("mps") == torch.device("mps")
    else:
        with pytest.raises(ValueError, match="MPS"):
            resolve_device("mps")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_dpm_matches_cpu_and_ddpm_is_seeded():
    model = ToyDenoiser()
    e, null = torch.ones(2, 768), torch.zeros(1, 768)
    cpu = sample_latents(model, e, null, steps=20, seed=13)
    gpu = sample_latents(model.to("mps"), e.to("mps"), null.to("mps"), steps=20, seed=13)
    assert torch.allclose(cpu, gpu.cpu(), atol=1e-4, rtol=1e-4)
    rng = torch.mps.get_rng_state().clone()
    a = sample_latents(model, e.to("mps"), null.to("mps"), sampler="ddpm", steps=250, seed=9)
    b = sample_latents(model, e.to("mps"), null.to("mps"), sampler="ddpm", steps=250, seed=9)
    assert torch.equal(torch.mps.get_rng_state(), rng)
    assert torch.equal(a, b) and torch.isfinite(a).all()


@pytest.fixture
def coco(tmp_path):
    source = tmp_path / "val2014"
    source.mkdir()
    images, captions, annotations = [], [], []
    for i in range(1, 5):
        name = f"COCO_val2014_{i:012d}.jpg"
        Image.new("RGB", (300 + i, 400), (20 * i, 40, 70)).save(source / name)
        images.append({"id": i, "file_name": name})
        captions += [{"image_id": i, "caption": f"first caption {i}"},
                     {"image_id": i, "caption": f"second caption {i}"}]
        annotations.append({"image_id": i, "bbox": [10, 20, 100, 180]})
    path = tmp_path / "captions_val2014.json"
    write_json(path, {"images": images, "annotations": captions})
    boxes = tmp_path / "person_keypoints_val2014.json"
    write_json(boxes, {"annotations": annotations})
    return path, source, boxes


@pytest.fixture
def manifest(coco, tmp_path):
    return prepare(coco[0], coco[1], tmp_path / "references", count=3, seed=17, preprocess="center-crop")


def test_frozen_subset_and_caption_alignment(coco, manifest, tmp_path):
    from eval.fetch_coco import select_images
    other = prepare(coco[0], coco[1], tmp_path / "other", count=3, seed=17, preprocess="center-crop")
    assert read_json(manifest) == read_json(other)
    rows = read_json(manifest)["records"]
    assert [r["image_id"] for r in rows] == [r["id"] for r in select_images(read_json(coco[0]), 3, 17)]
    assert len({row["image_id"] for row in rows}) == 3
    for row in rows:
        assert row["caption"] == f"first caption {row['image_id']}"
        with Image.open(manifest.parent / "real" / row["file"]) as image:
            assert image.size == (256, 256) and image.mode == "RGB"
    with pytest.raises(FileExistsError):
        prepare(coco[0], coco[1], manifest.parent, count=3, preprocess="center-crop")


def test_person_crop_uses_linda_preprocessing(coco, tmp_path):
    path = prepare(coco[0], coco[1], tmp_path / "people", count=2,
                   preprocess="person-crop", person_annotations=coco[2])
    assert read_json(path)["preprocess"] == "person-crop"
    with pytest.raises(ValueError, match="requires"):
        prepare(coco[0], coco[1], tmp_path / "bad", count=2, preprocess="person-crop")


def test_preparation_rejects_wrong_split_and_missing_images(coco, tmp_path):
    wrong = tmp_path / "captions_train2014.json"
    wrong.write_bytes(coco[0].read_bytes())
    with pytest.raises(ValueError, match="val2014"):
        prepare(wrong, coco[1], tmp_path / "bad", count=2, preprocess="center-crop")
    for path in coco[1].glob("*.jpg"):
        path.unlink()
    with pytest.raises(FileNotFoundError):
        prepare(coco[0], coco[1], tmp_path / "bad", count=2, preprocess="center-crop")


class FakePipeline:
    checkpoint_path = "/synthetic/ema.pt"
    checkpoint_sha256 = "synthetic"
    device = torch.device("cpu")
    training = {}
    null_condition = "synthetic_test_only"

    def __init__(self):
        self.calls = []
        self.fail_seed = None

    def generate(self, prompts, **settings):
        self.calls.append((prompts, settings))
        if settings["seed"] == self.fail_seed:
            raise RuntimeError("simulated interruption")
        g = torch.Generator().manual_seed(settings["seed"])
        return torch.rand(len(prompts), 3, 256, 256, generator=g)


def test_generation_resume_and_integrity(manifest, tmp_path):
    pipe = FakePipeline()
    pipe.fail_seed = 2
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="interruption"):
        generate_run(pipe, manifest, output, batch_size=2)
    assert read_json(output / "run.json")["status"] == "incomplete"
    with pytest.raises(ValueError, match="incomplete"):
        validate_run(output / "run.json")
    pipe.fail_seed = None
    pipe.calls = []
    # Simulate a crash after writing a temporary PNG; resume should recover it.
    (output / "images" / read_json(manifest)["records"][-1]["file"]).with_suffix(".tmp").write_bytes(b"partial")
    path = generate_run(pipe, manifest, output, batch_size=2, resume=True)
    assert len(pipe.calls) == 1 and pipe.calls[0][1]["seed"] == 2
    run, records = validate_run(path)
    assert run["status"] == "complete" and len(records) == 3
    assert (output / "preview.png").is_file()
    with pytest.raises(ValueError, match="Cannot resume"):
        generate_run(pipe, manifest, output, batch_size=2, resume=True, cfg_scale=3)
    image_path = output / "images" / records[0]["file"]
    Image.new("RGB", (256, 256)).save(image_path)
    with pytest.raises(ValueError, match="changed"):
        validate_run(path)


def test_metrics_match_subset_and_pair_captions(manifest, tmp_path, monkeypatch):
    path = generate_run(FakePipeline(), manifest, tmp_path / "run", count=2, batch_size=1)
    from cleanfid import fid
    import torchmetrics.multimodal
    seen = []

    def fake_fid(generated, references, **kwargs):
        gen = {p.name for p in Path(generated).glob("*.png")}
        real = {p.name for p in Path(references).glob("*.png")}
        assert gen == real and len(real) == 2
        assert kwargs["mode"] == "clean" and kwargs["model_name"] == "inception_v3"
        return 99.0

    class FakeScore:
        def __init__(self, model_name_or_path):
            self.name = model_name_or_path
            self.captions = []

        def to(self, device):
            return self

        def update(self, images, captions):
            assert images.dtype == torch.uint8 and images.shape == (1, 3, 256, 256)
            self.captions += captions

        def compute(self):
            seen.append((self.name, self.captions))
            return torch.tensor(25.0)

    monkeypatch.setattr(fid, "compute_fid", fake_fid)
    monkeypatch.setattr(torchmetrics.multimodal, "CLIPScore", FakeScore)
    result = read_json(compute_metrics(path, device="cpu", batch_size=1))
    assert result["fid_label"] == "FID-2" and result["fid"] == 99.0
    assert len(seen) == 2 and seen[0][1] == [r["caption"] for r in read_json(manifest)["records"][:2]]
    # Synthetic scores exercise plot/CSV plumbing, never reported as model results.
    figures = tmp_path / "figures"
    plot_results([path.parent / "metrics.json"], figures)
    assert all((figures / name).is_file() for name in ("results.csv", "cfg_metrics.png", "fid_vs_clip.png"))
    result["config"]["count"] = 3
    incompatible = tmp_path / "incompatible.json"
    write_json(incompatible, result)
    with pytest.raises(ValueError, match="incompatible"):
        plot_results([path.parent / "metrics.json", incompatible], tmp_path / "bad")


def test_sample_cli_and_cfg_sweep(manifest, tmp_path, monkeypatch):
    import sys
    import sample_t2i
    import eval.run
    pipe = FakePipeline()
    monkeypatch.setattr(TextToImagePipeline, "from_checkpoint", lambda *a, **k: pipe)
    output = tmp_path / "cli"
    monkeypatch.setattr(sys, "argv", ["sample_t2i.py", "--ckpt", "dummy.pt", "--prompt", "a cat",
                                    "--prompt", "a dog", "--output", str(output)])
    sample_t2i.main()
    assert (output / "preview.png").is_file()
    assert [row["prompt"] for row in read_json(output / "samples.json")["samples"]] == ["a cat", "a dog"]
    sweep = tmp_path / "sweep"
    monkeypatch.setattr(sys, "argv", ["eval.run", "sweep", "--ckpt", "dummy.pt", "--manifest", str(manifest),
                                    "--output", str(sweep), "--count", "2", "--cfg-scales", "1", "4"])
    eval.run.main()
    runs = [read_json(sweep / f"cfg_{s}" / "run.json") for s in (1, 4)]
    assert [r["config"]["cfg_scale"] for r in runs] == [1, 4]
    assert runs[0]["records"] == runs[1]["records"]  # dummy generator ignores CFG, same noise seeds


def test_convergence_plot(manifest, tmp_path):
    a = generate_run(FakePipeline(), manifest, tmp_path / "a", checkpoint_step=50000)
    b = generate_run(FakePipeline(), manifest, tmp_path / "b", checkpoint_step=100000)
    metrics_paths = []
    for run, fid_value in ((a, 150), (b, 130)):
        metrics = {"config": read_json(run)["config"], "fid": fid_value, "fid_label": "FID-3",
                   "fid_implementation": "synthetic-test", "metric_versions": {}, "clip_l14": 20,
                   "clip_b32": 21}
        destination = run.parent / "metrics.json"
        write_json(destination, metrics)
        metrics_paths.append(destination)
    plot_results(metrics_paths, tmp_path / "convergence", axis="step")
    assert (tmp_path / "convergence" / "convergence.png").is_file()


def test_run_rejects_extras_changed_manifest_and_misalignment(manifest, tmp_path):
    path = generate_run(FakePipeline(), manifest, tmp_path / "run")
    extra = path.parent / "images" / "extra.png"
    Image.new("RGB", (256, 256)).save(extra)
    with pytest.raises(ValueError, match="exactly"):
        validate_run(path)
    extra.unlink()
    data = read_json(path)
    data["records"][0]["caption"] = "wrong caption"
    write_json(path, data)
    with pytest.raises(ValueError, match="aligned"):
        validate_run(path)
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(ValueError, match="manifest has changed"):
        validate_run(path)


def test_load_ema_namespace_checkpoint_and_generate(model, tmp_path, monkeypatch):
    """Use the actual full DiT-S/4 and schedulers with lightweight CLIP/VAE substitutes."""
    checkpoint = tmp_path / "dummy.pt"
    with torch.no_grad():
        model.y_embedder.proj[0].weight.fill_(0.123)
    state = model.state_dict()
    raw = {k: v.clone() for k, v in state.items()}
    raw["y_embedder.proj.0.weight"].zero_()
    torch.save({"ema": state, "model": raw, "args": Namespace(model="DiT-S/4"), "step": 5}, checkpoint)
    import transformers
    import diffusers

    class Tokens(dict):
        def to(self, device):
            return self

    class Tokenizer:
        def __call__(self, prompts, **kwargs):
            assert kwargs["max_length"] == 77 and kwargs["padding"] == "max_length"
            return Tokens(input_ids=torch.ones(len(prompts), 77, dtype=torch.long))

    class Encoder(nn.Module):
        def forward(self, input_ids):
            return SimpleNamespace(pooler_output=torch.ones(len(input_ids), 768))

    class VAE(nn.Module):
        def decode(self, z):
            return SimpleNamespace(sample=torch.nn.functional.interpolate(z[:, :3], size=(256, 256)))

    monkeypatch.setattr(transformers.CLIPTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    monkeypatch.setattr(transformers.CLIPTextModel, "from_pretrained", lambda *a, **k: Encoder())
    monkeypatch.setattr(diffusers.AutoencoderKL, "from_pretrained", lambda *a, **k: VAE())
    pipe = TextToImagePipeline.from_checkpoint(checkpoint, device="cpu", local_files_only=True)
    assert pipe.checkpoint_sha256 == sha256(checkpoint)
    assert torch.allclose(pipe.model.y_embedder.proj[0].weight, torch.full_like(pipe.model.y_embedder.proj[0].weight, 0.123))
    for sampler in ("dpm", "ddpm"):
        images = pipe.generate(["a dog"], sampler=sampler, steps=5)
        assert images.shape == (1, 3, 256, 256)
        assert torch.isfinite(images).all() and images.min() >= 0 and images.max() <= 1
    state["y_embedder.null_embedding"].zero_()
    torch.save({"ema": state}, checkpoint)
    with pytest.raises(ValueError, match="nonzero"):
        TextToImagePipeline.from_checkpoint(checkpoint, device="cpu", local_files_only=True)
    with pytest.warns(UserWarning, match="Preserving"):
        zero_pipe = TextToImagePipeline.from_checkpoint(checkpoint, device="cpu", local_files_only=True,
                                                       allow_zero_null=True)
    assert zero_pipe.null_condition == "zeros_from_training_checkpoint"
    assert torch.count_nonzero(zero_pipe.model.y_embedder.null_embedding) == 0


def test_record_actual_training_step(manifest, tmp_path):
    pipe = FakePipeline()
    pipe.training = {"step": 150000, "weights": "ema", "args": {"global_batch_size": 128}}
    run = generate_run(pipe, manifest, tmp_path / "actual_step")
    config = read_json(run)["config"]
    assert config["checkpoint_step"] == 150000
    assert config["training"]["args"]["global_batch_size"] == 128
    with pytest.raises(ValueError, match="does not match"):
        generate_run(pipe, manifest, tmp_path / "wrong_step", checkpoint_step=400000)
