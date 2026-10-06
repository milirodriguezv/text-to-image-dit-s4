"""
Shared pytest fixtures for the DiT-S/4 text-conditioning tests.
Puts base_code/ on sys.path so `models` and `diffusion` import from any working directory.
"""
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

BASE_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_CODE not in sys.path:
    sys.path.insert(0, BASE_CODE)

from models import DiT_models  # noqa: E402


def build_model(null_path=None, **kwargs):
    """DiT-S/4 as used in training (input 32x32 latents, 768D text, 15% text dropout)."""
    return DiT_models["DiT-S/4"](input_size=32, text_dim=768, null_path=null_path, **kwargs)


@pytest.fixture
def null_npy(tmp_path):
    """Synthetic ∅ saved like extract_features.py would (fp16, shape (1, 768)). No CLIP needed."""
    rng = np.random.default_rng(0)
    null = rng.standard_normal((1, 768)).astype(np.float16)
    path = tmp_path / "null_empty_string.npy"
    np.save(path, null)
    return path, null


@pytest.fixture
def model(null_npy):
    """Freshly initialized model (adaLN-Zero: output is identically zero)."""
    torch.manual_seed(0)
    return build_model(null_path=str(null_npy[0]))


@pytest.fixture
def random_model(model):
    """
    Model whose zero-initialized layers are re-initialized randomly (normal std 0.05).
    Needed because at init adaLN-Zero makes the output identically zero for ANY y, so tests
    of conditioning/CFG would pass trivially. Only done in tests; models.py keeps the zero-init.
    """
    torch.manual_seed(1)
    zero_layers = [b.adaLN_modulation[-1] for b in model.blocks]
    zero_layers += [model.final_layer.adaLN_modulation[-1], model.final_layer.linear]
    with torch.no_grad():
        for layer in zero_layers:
            nn.init.normal_(layer.weight, std=0.05)
            nn.init.normal_(layer.bias, std=0.05)
    return model.eval()
