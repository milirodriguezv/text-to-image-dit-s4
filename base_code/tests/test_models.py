"""
Unit tests for the text-conditioned DiT-S/4 (adaLN-Zero-Text) in models.py.

Covers: parameter count, output shape, adaLN-Zero zero output, the frozen ∅ buffer (loading,
state_dict, deepcopy/EMA), text dropout for CFG (rate, train/eval, force_drop_ids), that text
changes the output, forward_with_cfg (4 guided ε channels, unguided Σ), compatibility with
diffusion/ (training_losses, p_sample_loop), bf16 autocast and gradient flow to TextEmbedder.

CPU only, synthetic inputs (no CLIP, no data, no network). Run:
    cd base_code && ../.venv/bin/python -m pytest tests -q
    # or, from the repo root:  .venv/bin/python -m pytest base_code/tests -q
"""
import copy

import numpy as np
import pytest
import torch

from conftest import build_model
from diffusion import create_diffusion

ATOL = 1e-5  # batching (B vs 2B) can introduce ~1e-7 float noise


def _inputs(n=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, 32, 32, generator=g)
    t = torch.randint(0, 1000, (n,), generator=g)
    e = torch.randn(n, 768, generator=g)
    return x, t, e


def _null(model, n):
    return model.y_embedder.null_embedding.expand(n, -1)


# ---------------------------------------------------------------------------- shapes / init

def test_param_count(model):
    assert sum(p.numel() for p in model.parameters()) == 33_003_776


def test_output_shape(model):
    x, t, e = _inputs()
    assert t.dtype == torch.long
    with torch.no_grad():
        out = model(x, t, e)
    assert out.shape == (2, 8, 32, 32)


def test_zero_output_at_init(model):
    x, t, e = _inputs()
    with torch.no_grad():
        out = model(x, t, e)
    assert torch.all(out == 0)


# ---------------------------------------------------------------------------- ∅ buffer

def test_null_is_buffer_and_restored_from_state_dict(model):
    key = "y_embedder.null_embedding"
    assert key in model.state_dict()
    assert key not in dict(model.named_parameters())
    # A model built without null_path (∅ = zeros) gets ∅ back from the checkpoint.
    other = build_model(null_path=None)
    assert torch.all(other.y_embedder.null_embedding == 0)
    other.load_state_dict(model.state_dict())
    assert torch.equal(other.y_embedder.null_embedding, model.y_embedder.null_embedding)


@pytest.mark.parametrize("shape", [(1, 768), (768,)])
def test_null_loaded_from_npy(tmp_path, shape):
    null = np.random.default_rng(1).standard_normal(shape).astype(np.float16)
    path = tmp_path / "null.npy"
    np.save(path, null)
    m = build_model(null_path=str(path))
    buf = m.y_embedder.null_embedding
    assert buf.shape == (1, 768)
    assert buf.dtype == torch.float32
    assert torch.equal(buf, torch.from_numpy(null.astype(np.float32)).reshape(1, 768))


def test_null_wrong_size_raises_and_none_is_zeros(tmp_path):
    path = tmp_path / "bad_null.npy"
    np.save(path, np.zeros((1, 512), dtype=np.float16))
    with pytest.raises(AssertionError):
        build_model(null_path=str(path))
    m = build_model(null_path=None)
    assert m.y_embedder.null_embedding.shape == (1, 768)
    assert torch.all(m.y_embedder.null_embedding == 0)


# ---------------------------------------------------------------------------- text dropout

def test_dropout_rate_is_15_percent(model):
    emb = model.y_embedder
    assert emb.dropout_prob == 0.15
    torch.manual_seed(0)
    n = 100_000
    e = torch.randn(n, 768) + 10.0  # far from ∅, so no row equals ∅ by chance
    out = emb.token_drop(e)
    dropped = (out == emb.null_embedding).all(dim=1)
    rate = dropped.float().mean().item()
    assert abs(rate - 0.15) <= 0.01, rate
    assert torch.equal(out[~dropped], e[~dropped])  # non-dropped rows untouched


def test_no_dropout_in_eval(model):
    model.eval()
    emb = model.y_embedder
    torch.manual_seed(0)
    e = torch.randn(64, 768)
    with torch.no_grad():
        assert torch.equal(emb(e, train=False), emb.proj(e))
        # dropout_prob = 0 with train=True -> no dropout either
        emb.dropout_prob = 0.0
        assert torch.equal(emb(e, train=True), emb.proj(e))


def test_train_vs_eval_through_forward(random_model):
    model = random_model
    x, t, e = _inputs(n=16)
    with torch.no_grad():
        model.eval()
        a, b = model(x, t, e), model(x, t, e)
        assert torch.equal(a, b)  # eval: deterministic (no dropout)
        model.train()
        torch.manual_seed(0)
        c = model(x, t, e)
        d = model(x, t, e)
    # train: random ∅ replacement makes repeated calls differ (fixed seed -> deterministic test)
    assert not torch.allclose(c, d)


@pytest.mark.parametrize("train", [True, False])
def test_force_drop_ids(model, train):
    emb = model.y_embedder
    torch.manual_seed(0)
    e = torch.randn(2, 768)
    with torch.no_grad():
        out = emb(e, train=train, force_drop_ids=torch.tensor([1, 0]))
        assert torch.allclose(out[0], emb.proj(emb.null_embedding)[0], atol=ATOL)
        assert torch.allclose(out[1], emb.proj(e[1:2])[0], atol=ATOL)


def test_text_changes_output(random_model):
    model = random_model
    x, t, e = _inputs()
    e2 = torch.randn(2, 768, generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        out1 = model(x, t, e)
        out2 = model(x, t, e2)
        out_null = model(x, t, _null(model, 2))
    assert not torch.allclose(out1, out2, atol=1e-3)
    assert not torch.allclose(out1, out_null, atol=1e-3)


# ---------------------------------------------------------------------------- CFG

def test_cfg_matches_manual_formula_all_4_channels(random_model):
    model, s, n = random_model, 4.0, 2
    z, t, e = _inputs(n)
    with torch.no_grad():
        cond = model(z, t, e)
        uncond = model(z, t, _null(model, n))
        out = model.forward_with_cfg(torch.cat([z, z]), torch.cat([t, t]),
                                     torch.cat([e, _null(model, n)]), s)
    manual = uncond[:, :4] + s * (cond[:, :4] - uncond[:, :4])
    assert out.shape == (2 * n, 8, 32, 32)
    assert torch.allclose(out[:n, :4], manual, atol=ATOL)
    # channel 3 is guided too (the original repo only guided channels 0-2)
    assert not torch.allclose(out[:n, 3], cond[:, 3], atol=1e-3)


def test_cfg_scale_1_and_0_halves_sigma_and_ignored_x(random_model):
    model, n = random_model, 2
    z, t, e = _inputs(n)
    t2, y = torch.cat([t, t]), torch.cat([e, _null(model, n)])
    with torch.no_grad():
        cond = model(z, t, e)
        uncond = model(z, t, _null(model, n))
        out1 = model.forward_with_cfg(torch.cat([z, z]), t2, y, 1.0)
        out0 = model.forward_with_cfg(torch.cat([z, z]), t2, y, 0.0)
        # second half of x is ignored (only the first half is used, duplicated)
        out_other = model.forward_with_cfg(torch.cat([z, torch.randn_like(z)]), t2, y, 1.0)
    assert torch.allclose(out1[:n, :4], cond[:, :4], atol=ATOL)    # s=1 -> conditional
    assert torch.allclose(out0[:n, :4], uncond[:, :4], atol=ATOL)  # s=0 -> unconditional
    assert torch.equal(out1[:n, :4], out1[n:, :4])                 # both halves identical
    # Σ channels unguided: cond half from (z, e), uncond half from (z, ∅)
    assert torch.allclose(out1[:n, 4:], cond[:, 4:], atol=ATOL)
    assert torch.allclose(out1[n:, 4:], uncond[:, 4:], atol=ATOL)
    assert torch.equal(out_other, out1)


# ---------------------------------------------------------------------------- diffusion/

def test_training_losses_and_p_sample_loop(random_model):
    model, n = random_model, 2
    model.train()
    z0, t, e = _inputs(n)
    diffusion = create_diffusion(timestep_respacing="")
    loss_dict = diffusion.training_losses(model, z0, t, dict(y=e))
    for k in ("loss", "mse", "vb"):
        assert loss_dict[k].shape == (n,)
        assert torch.isfinite(loss_dict[k]).all()

    model.eval()
    torch.manual_seed(0)
    y = torch.cat([e, _null(model, n)])
    with torch.no_grad():
        samples = create_diffusion("5").p_sample_loop(
            model.forward_with_cfg, (2 * n, 4, 32, 32), torch.randn(2 * n, 4, 32, 32),
            clip_denoised=False, model_kwargs=dict(y=y, cfg_scale=4.0), progress=False, device="cpu",
        )
    assert samples.shape == (2 * n, 4, 32, 32)
    assert torch.isfinite(samples).all()


# ---------------------------------------------------------------------------- EMA / precision / grads

def test_deepcopy_keeps_null_buffer(model):
    ema = copy.deepcopy(model)
    assert torch.equal(ema.y_embedder.null_embedding, model.y_embedder.null_embedding)
    assert "y_embedder.null_embedding" in dict(ema.named_buffers())
    assert "y_embedder.null_embedding" not in dict(ema.named_parameters())


def test_bf16_autocast_cpu(random_model):
    x, t, e = _inputs()
    try:
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            out = random_model(x, t, e)
    except RuntimeError as err:
        pytest.skip(f"CPU bf16 autocast unsupported: {err}")
    assert out.shape == (2, 8, 32, 32)
    assert torch.isfinite(out.float()).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bf16_autocast_cuda(random_model):
    model = random_model.cuda()
    x, t, e = (a.cuda() for a in _inputs())
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(x, t, e)
    assert out.shape == (2, 8, 32, 32)
    assert torch.isfinite(out.float()).all()


def test_gradients_reach_text_embedder(model):
    """
    adaLN-Zero: at step 0 only final_layer.linear gets gradient (everything upstream is multiplied
    by zero weights). Step 1 updates the adaLN layers; by step 2 the gradient reaches TextEmbedder.
    """
    torch.manual_seed(0)
    model.train()
    diffusion = create_diffusion(timestep_respacing="")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    z0, _, e = _inputs(n=4)
    proj_grad_nonzero = []
    for step in range(3):
        t = torch.randint(0, 1000, (4,))
        loss = diffusion.training_losses(model, z0, t, dict(y=e))["loss"].mean()
        opt.zero_grad()
        loss.backward()
        grads = {k: p.grad for k, p in model.named_parameters() if p.grad is not None}
        nonzero = {k for k, g in grads.items() if g.abs().sum() > 0}
        if step == 0:
            assert nonzero == {"final_layer.linear.weight", "final_layer.linear.bias"}
        proj_grad_nonzero.append("y_embedder.proj.0.weight" in nonzero)
        opt.step()
    assert proj_grad_nonzero == [False, False, True]
