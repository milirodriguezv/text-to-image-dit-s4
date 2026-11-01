# CLAUDE.md — Text-to-Image DiT-S/4 (adaLN-Zero-Text)

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context
- Goal: adapt the class-conditional DiT (Peebles & Xie, 2023; repo `facebookresearch/DiT`) into a
  **text-conditioned** latent diffusion model, DiT-S/4, trained on MS-COCO Captions.
- Conditioning: **Variant A, adaLN-Zero-Text (global modulation)**. The frozen CLIP pooled text
  vector `e_pooled` is projected and summed with the timestep embedding. Variant B
  (cross-attention with `E_seq`) is out of scope unless explicitly requested.
- Status: the 150K run and its evaluation are finished and integrated on `main`.


## Terminology (use exactly these names)
`z_0` clean latent · `z_t` noisy latent · `z_T` initial noise · `ε` / `ε_t` true noise ·
`ε_θ` predicted noise · `Σ_θ` predicted variance · `e_pooled` (768D CLIP [EOS] vector) ·
`E_seq` (77×768, Variant B only) · `∅` null text embedding · `t_emb`, `y_emb`, `c = t_emb + y_emb` ·
adaLN-Zero · DiT-S/4 · p=4 · d=384 · T=64 tokens · s = CFG scale.

## Original specification
These are the planned values. The final run deviates from them; see "Final run" below,
which prevails. The detailed report notes were removed from the tree; they remain
in git history at commit `edaed2f` (`docs/REPORT_NOTES.md`, `docs/eval_README.md`).

| Item | Value |
|---|---|
| Image / latent | 256×256×3 → 4×32×32 (f=8), scale factor 0.18215 |
| VAE | `stabilityai/sd-vae-ft-ema` (frozen), used for BOTH encoding and decoding |
| Text encoder | `openai/clip-vit-large-patch14`, `CLIPTextModel.pooler_output` (frozen), max_length 77 |
| Model | DiT-S/4: depth 12, hidden 384, heads 6, patch 4, 64 tokens, `learn_sigma=True` (8 output channels) |
| Text projection | `TextEmbedder`: Linear(768→384) → SiLU → Linear(384→384) |
| ∅ | Frozen CLIP embedding of `""` (register_buffer, not trained) |
| Text dropout (CFG) | 15% |
| Loss | Hybrid (repo default): MSE on ε_θ + vb (VLB, bits per dim) on Σ_θ; the code adds `mse + vb` with no 0.001 factor |
| Optimizer | AdamW, lr 1e-4, weight_decay 0, global batch 256, no warmup |
| EMA | decay 0.9999, used for all sampling/evaluation |
| Training length | 400K steps (confirm with measured it/s; ~33–37 h estimated on RTX 3090) |
| Precision | bf16 autocast around the model forward only (loss in fp32), after 500-step check vs fp32 |
| Grad clipping | None (as in paper); log grad norm every 100 steps |
| Diffusion | 1000 steps, linear β 1e-4 → 0.02, t ∈ {0,…,999} |
| Samplers | Default DPM-Solver++ (diffusers, 20 steps, order 2); DDPM 250 steps (repo) when requested |
| CFG | Applied to all 4 ε channels; sweep s ∈ {1.0, 1.5, 2, 3, 4, 5} |
| Data | Train: COCO train2014 (82,783 imgs, first 5 captions each). Eval: 10K imgs/captions from val2014 |
| Metrics | FID-10K via clean-fid (FID-2K for intermediate ckpts); CLIP Score ViT-L/14 (primary) + ViT-B/32 |


## Final run
| Item | Specification | Final run |
|---|---|---|
| Training images | train2014, 82,783 | 45,174 images with a person box and 5 captions |
| Crop | Center crop | Person boxes + 20% margin, made square, clipped to the image (non-square at borders), resized to 256 |
| Global batch | 256 | 128 |
| Training length | 400K steps | 150K steps |
| Hardware | RTX 3090 | CPU with AVX-512 BF16, ~21 h |
| ∅ | Frozen CLIP(`""`) | All zeros: `train.py` does not pass `null_path` |
| Data loading | `DistributedSampler`, `torchrun` | Single process, `DataLoader(shuffle=True)` |
| Evaluation | 10K val2014, center crop | As specified, not filtered by people; FID-2K at 150K only; Apple Silicon MPS |

Checkpoint: `last.pt`, EMA at step 150,000, SHA-256 `f693fd73…`; `eval.fetch_checkpoint` downloads it.

## Original-code facts to rely on (verified in `dit_base/` and the original repo)
- **`training_losses`** (`diffusion/gaussian_diffusion.py`) returns per-sample `[N]` tensors under
  `"loss"`, `"mse"` and `"vb"`. Take `.mean()` of each for logging.
  - `loss_type` is plain `MSE`, not `RESCALED_MSE`, so `vb` is not rescaled.
  - It asserts that the model output has `2*C` channels, so `learn_sigma` must stay `True`.
  - It calls the model internally as `model(x_t, t, **model_kwargs)`, so the conditioning kwarg must
    be named `y`.
- **bf16 with `diffusion/` frozen:** because the model call happens inside `training_losses`, "autocast
  on the model forward only" has to be done with a thin wrapper, for example a callable that runs
  `model(x, t, **kw)` under `torch.autocast(...)` and returns `.float()`. Pass that wrapper to
  `training_losses`. Do NOT wrap the whole `training_losses` call.
- **Respaced timesteps:** `SpacedDiffusion` (`respace.py`) wraps the model in `_WrappedModel`, which
  maps respaced indices back to the original 0–999 indices, so DiT always sees original-scale `t`.
  - diffusers' DPM-Solver++ timesteps are already on the original scale. Call the model directly with
    `t` expanded to the batch as a long tensor.
  - `create_diffusion` also accepts `"ddimN"` strings.
- **`LabelEmbedder`**, the pattern `TextEmbedder` should mirror:
  - Dropout applies only when `(train and dropout_prob > 0) or force_drop_ids is not None`.
  - `DiT.forward` passes `self.training`, so `model.train()` / `model.eval()` switch CFG dropout on
    and off.
- **`forward_with_cfg`** takes a doubled batch but only uses the first half of `x`
  (`half = x[:len(x)//2]`, duplicated). It returns guided ε for both halves plus the unguided Σ
  channels. Callers keep `samples.chunk(2)[0]`.
- **`src/train.py` internals:**
  - Single process on CUDA or CPU, no MPS. Run with `--results-dir outputs/runs`; new dirs are `NNN-<model>/`,
    where NNN is the count of existing entries.
  - Checkpoint dict is `{"model", "ema", "opt", "args", "step"}`: `last.pt` every 10K steps, a permanent one every 50K.
  - `--use-amp auto` enables bf16 on CUDA or on a CPU with AVX-512 BF16.
  - `update_ema` walks `named_parameters()` only, so buffers such as ∅ are copied by `deepcopy` and never
    EMA-updated. That is fine because ∅ is constant.
- **`download.find_model(path)`** returns `checkpoint["ema"]` when that key exists. Reuse it in
  `sample_t2i.py` and `eval/`.
  - `src/train.py` stores `vars(args)` under `"args"`. `find_model` passes `weights_only=False` and also accepts an
    argparse `Namespace`, which `torch.load` rejects by default on PyTorch ≥2.6. Only load trusted files.
- `models.py` imports `PatchEmbed`, `Attention` and `Mlp` from `timm.models.vision_transformer`.

## Repository map
| Path | Status | Role |
|---|---|---|
| `dit_base/models.py` | Adapted | `TextEmbedder` replaces `LabelEmbedder`; `forward_with_cfg` fixed |
| `dit_base/diffusion/` | DO NOT EDIT | Schedule, `q_sample`, `training_losses` (hybrid loss), DDPM/DDIM loops |
| `dit_base/download.py`, `LICENSE.txt` | Adapted / keep | `find_model`; original license, required |
| `src/train.py` | New | Precomputed features, `y = e_pooled`, resume, max-steps, periodic sampling |
| `src/extract_features.py`, `createCSV.py`, `make_null_embedding.py`, `text_utils.py` | New | Offline VAE + CLIP encoding to disk |
| `src/dataset_t2i.py` | New | `CocoLatentDataset` over memory-mapped arrays |
| `src/sample_t2i.py` | New | Text-prompt sampling: DPM-Solver++ 20 / DDPM 250, CFG |
| `src/eval/`, `src/analyze_evaluation.py`, `src/report_evaluation.py`, `src/verify_evaluation.py` | New | Generation, FID (clean-fid), CLIP Score, analyses |
| `tests/` | New | Tests |
| `notebooks/` | New | Tutorial notebook required by the course brief, stored with outputs |
| `outputs/` | Results | `training/` (log, grids), `evaluation/dit_s4_150k/`; new runs in `outputs/runs/`, not versioned |
| `text_conditioned_dit_report/` | Report | `main.tex`, `references.bib`, style file and copies of the report figures; compile from this folder |

Run from the repo root with `export PYTHONPATH="$PWD/dit_base:$PWD/src"`. Exact commands are in `README.md`.

## Pipeline as built
1. **Environment:** Python 3.11, `pip install -r requirements.txt` (evaluation versions; training versions
   were not recorded). Tests: `python -m pytest tests -q`, expected 50 passed and 1 skipped without CUDA.
2. **Data:** `coco/` with `train2014/`, `val2014/`, and captions and `person_keypoints` for train and val.
   Never train on train2017, which contains most of val2014.
3. **Features** (`extract_features.py`, `createCSV.py`, `make_null_embedding.py`): write to `features/train/`
   `latents.npy` (N, 2, 8, 32, 32; mean + std, unscaled, original + flip), `e_pooled.npy` (N, 5, 768),
   `image_ids.npy`, `captions.csv` and `null_empty_string.npy`. Extraction is resumable and CPU only. Only images
   with people are kept. `E_seq` is not stored.
4. **Dataset** (`dataset_t2i.py`): memory-mapped; each item draws a flip and a caption with `torch.randint`, and
   returns `z_0 = (mean + std * randn) * 0.18215` (applied once) and `e_pooled`.
5. **Model** (`dit_base/models.py`): `TextEmbedder` (`proj`, `null_embedding` buffer, `token_drop`);
   `forward_with_cfg` guides all 4 ε channels. 33,003,776 parameters and zero output at init. `null_path=None`
   leaves ∅ in zeros.
6. **Training** (`src/train.py`): `AutocastModelWrapper` applies bf16 to the model forward only; logs loss, mse, vb
   and grad norm; every 10K steps saves a 4×2 grid with 8 fixed prompts (EMA, DPM-Solver++ 20, s = 4, seed 0).
7. **Sampling** (`src/sample_t2i.py`): DPM-Solver++ 20 or DDPM 250 from EMA weights, taking ∅ from the checkpoint.
   This run needs `--allow-zero-null`.
8. **Evaluation** (`src/eval/`): `prepare` freezes the manifest, `execute_plan` runs FID-2K, the six FID-10K CFG
   values and DDPM 250. `run` handles a single configuration. `analyze_evaluation.py` computes real-image CLIP and
   the with/without-people groups, and needs the full PNGs. Never mix FID implementations or sample counts.

## Known problems, left unfixed to preserve this run
- ∅ in zeros (see "Final run"). Fix `train.py` before a run that should follow the specification.
- `update_ema(ema, model, decay=0)` (`src/train.py:244`) overwrites the EMA on resume. Fix before using `--resume`.
- `code_sha256` differs from the delivered runs since the 01/11 reorganization.

## Rules for Claude when working in this repo
- Do not edit `diffusion/`. Keep paper code in `dit_base/` and new code in `src/`.
- Keep the adaLN-Zero zero-initialization intact.
- Any new deviation from the specification must be flagged and recorded in "Final run" above.
- When a hyperparameter isn't specified here, say so instead of assuming.
- Update this file, `README.md` and the report (`text_conditioned_dit_report/main.tex`) whenever code behavior or results change.
