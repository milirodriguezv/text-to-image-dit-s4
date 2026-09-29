# CLAUDE.md — Text-to-Image DiT-S/4 (adaLN-Zero-Text)

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context
- Goal: adapt the class-conditional DiT (Peebles & Xie, 2023; repo `facebookresearch/DiT`) into a
  **text-conditioned** latent diffusion model, DiT-S/4, trained on MS-COCO Captions.
- Conditioning: **Variant A, adaLN-Zero-Text (global modulation)**. The frozen CLIP pooled text
  vector `e_pooled` is projected and summed with the timestep embedding. Variant B
  (cross-attention with `E_seq`) is out of scope unless explicitly requested.
- Phase 2 (code development + training): 23/09 – 15/10. Target: full training run started by ~03/10.


## Terminology (use exactly these names)
`z_0` clean latent · `z_t` noisy latent · `z_T` initial noise · `ε` / `ε_t` true noise ·
`ε_θ` predicted noise · `Σ_θ` predicted variance · `e_pooled` (768D CLIP [EOS] vector) ·
`E_seq` (77×768, Variant B only) · `∅` null text embedding · `t_emb`, `y_emb`, `c = t_emb + y_emb` ·
adaLN-Zero · DiT-S/4 · p=4 · d=384 · T=64 tokens · s = CFG scale.

## Fixed specification
| Item | Value |
|---|---|
| Image / latent | 256×256×3 → 4×32×32 (f=8), scale factor 0.18215 |
| VAE | `stabilityai/sd-vae-ft-ema` (frozen), used for BOTH encoding and decoding |
| Text encoder | `openai/clip-vit-large-patch14`, `CLIPTextModel.pooler_output` (frozen), max_length 77 |
| Model | DiT-S/4: depth 12, hidden 384, heads 6, patch 4, 64 tokens, `learn_sigma=True` (8 output channels) |
| Text projection | `TextEmbedder`: Linear(768→384) → SiLU → Linear(384→384) |
| ∅ | Frozen CLIP embedding of `""` (register_buffer, not trained) |
| Text dropout (CFG) | 15% |
| Loss | Hybrid (repo default): MSE on ε_θ + vb (VLB) on Σ_θ, i.e. L_simple + 0.001·L_vlb |
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


## Original-code facts to rely on (verified in `base_code/`)
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
- **`train.py` internals:**
  - Checkpoint dict is `{"model", "ema", "opt", "args"}`. Resume must add `"step"`, and should reuse
    the existing experiment dir. New dirs are `results/NNN-<model>/`, where NNN is the count of existing
    entries.
  - Per-rank seed is `global_seed * world_size + rank`.
  - Only rank 0 logs (`create_logger(None)` on the other ranks).
  - `update_ema` walks `named_parameters()` only, so buffers such as the frozen ∅ are copied by
    `deepcopy` and never EMA-updated. That is fine because ∅ is constant.
  - TF32 is enabled at the top of `train.py` and `sample.py`.
- **`download.find_model(path)`** returns `checkpoint["ema"]` when that key exists. Reuse it in
  `sample_t2i.py` and `eval/`.
  - `environment.yml` pins only `pytorch>=1.13`. On PyTorch ≥2.6, `torch.load` defaults to
    `weights_only=True` and fails on our checkpoints, which contain an argparse `Namespace` under
    `"args"`. Pass `weights_only=False`, or save `vars(args)`.
- **Class-conditional leftovers:** once `models.py` is modified, `sample.py`, `sample_ddp.py`,
  `download.py`'s pretrained XL/2 weights and `run_DiT.ipynb` no longer work. They hardcode
  `y_null = 1000`, `--num-classes` and class-label lists. Treat them as reference only.
- `models.py` imports `PatchEmbed`, `Attention` and `Mlp` from `timm.models.vision_transformer`.

## Repository map
All paths are inside `base_code/`.
| File | Status | Role |
|---|---|---|
| `models.py` | MODIFY | Replace `LabelEmbedder` with `TextEmbedder`; fix `forward_with_cfg` |
| `diffusion/` | DO NOT EDIT | Schedule, `q_sample`, `training_losses` (hybrid loss), DDPM/DDIM loops |
| `train.py` | MODIFY | Load precomputed features, `y = e_pooled`, resume, max-steps, periodic sampling |
| `extract_features.py` | NEW | Offline VAE + CLIP encoding to disk |
| `datasets_t2i.py` | NEW | `CocoLatentDataset` over memory-mapped arrays |
| `sample_t2i.py` | NEW | Text-prompt sampling: DPM-Solver++ 20 / DDPM 250, CFG |
| `eval/` | NEW | 10K generation, FID-10K (clean-fid), CLIP Score (torchmetrics) |
| `sample.py`, `sample_ddp.py` | REFERENCE | Adapt `sample_ddp.py` batching/npz logic for eval |

## Implementation steps (in order)

### 1. Environment
- `conda env create -f environment.yml && conda activate DiT`
- `pip install -U transformers diffusers torchmetrics clean-fid`
- Windows: `train.py` uses NCCL → use WSL2 or switch the backend to `gloo`.
- Work on branch `t2i`, so all changes to the original repo stay visible as a diff.

### 2. Data
- Download COCO `train2014/`, `val2014/`, `annotations/captions_train2014.json`,
  `annotations/captions_val2014.json`.
- Never train on train2017 (it contains most of val2014, which would leak the eval set).

### 3. `extract_features.py` (run once)
- Preprocess: ADM `center_crop_arr` to 256, `ToTensor`, `Normalize([0.5]*3, [0.5]*3)` → [-1, 1].
- For each image: original + horizontal flip → `vae.encode(x).latent_dist`.
  Store `cat([mean, std])` (8 ch) in fp16.
- For each image: first 5 captions → CLIP tokenizer (`padding="max_length"`, `max_length=77`,
  `truncation=True`) → `pooler_output` (768) in fp16.
- Outputs (`features/train/`):
  - `latents.npy` (N, 2, 8, 32, 32)
  - `e_pooled.npy` (N, 5, 768)
  - `null_empty_string.npy` (1, 768)
- `shuffle=False`; row i of every array = image i. Assert ≥5 captions per image.
- Do NOT store `E_seq` (~50 GB; Variant B only).
- For val2014: store `e_pooled` for the 10K eval captions, plus the image path/caption list.
- CHECK: decode row 0's stored `mean` directly with `vae.decode(mean)` (it is stored unscaled) and
  compare visually with the cropped original.

### 4. `datasets_t2i.py`
- `np.load(..., mmap_mode="r")`.
- Per item: random flip f ∈ {0,1} and caption k ∈ {0..4}, drawn with `torch.randint`
  (not np.random).
- `z_0 = (mean + std * randn) * 0.18215`; returns `(z_0 [4,32,32] float32, e_pooled [768] float32)`.

### 5. `models.py`
- Add `TextEmbedder(text_dim=768, hidden_size, dropout_prob, null_path)`:
  - `proj` = Linear → SiLU → Linear.
  - `register_buffer("null_embedding", CLIP("") vector)`.
  - `token_drop` replaces rows with ∅ where `rand < dropout_prob` (or where `force_drop_ids == 1`).
  - `forward(e, train, force_drop_ids=None)`.
- `DiT.__init__`: replace `num_classes` with `text_dim=768`;
  `self.y_embedder = TextEmbedder(...)`; instantiate with `class_dropout_prob=0.15`.
- `initialize_weights`: delete the `embedding_table` init line; add
  `nn.init.normal_(std=0.02)` on `proj[0].weight` and `proj[2].weight`.
  KEEP the adaLN-Zero and final-layer zero-inits.
- `DiT.forward(x, t, y)`: unchanged (`c = t_embedder(t) + y_embedder(y, self.training)`).
- `forward_with_cfg`: split `model_out[:, :self.in_channels]` (all 4 channels, not the repo's 3);
  `y = cat([e_prompt, e_null])`, conditional half first.
- CHECK:
  - Parameter count ≈ 33.4M.
  - `model(randn(2,4,32,32), randint(0,1000,(2,)), randn(2,768))` → shape (2,8,32,32), all zeros at init.

### 6. `train.py`
- Remove the VAE load and the on-the-fly `vae.encode`; use `CocoLatentDataset`.
- Keep `DistributedSampler`, `pin_memory`, `drop_last`.
- Model: `DiT_models["DiT-S/4"](input_size=32, text_dim=768, class_dropout_prob=0.15)`.
- `diffusion = create_diffusion(timestep_respacing="")` (unchanged).
- Step:
  - `t = randint(0, 1000)`
  - `loss_dict = diffusion.training_losses(model, z0, t, dict(y=e))`
  - `loss = loss_dict["loss"].mean()`
- Log `loss`, `mse`, `vb`, grad norm (`clip_grad_norm_(params, inf)`), steps/sec every 100 steps.
- bf16: `torch.autocast("cuda", dtype=torch.bfloat16)` around the model forward only.
- Add:
  - `--max-steps 400000`
  - `--resume <ckpt>` (restores model, ema, opt, step)
  - `last.pt` every 10K steps
  - permanent `{step:07d}.pt` every 50K steps
- Periodic sampling every 10K steps: EMA model, 8 fixed prompts, fixed seed 0,
  DPM-Solver++ 20 steps, s = 4.0 → `samples/{step:07d}.png` (4×2 grid).
- Launch: `torchrun --nnodes=1 --nproc_per_node=1 train.py --model DiT-S/4 --feature-path features/train ...`

### 7. Sanity checks BEFORE the full run
1. VAE round trip (step 3 CHECK).
2. Model shape/zero-output test (step 5 CHECK).
3. Overfit test: 64 images, 1 caption each, no flip, dropout 0, batch 64, ~2K steps.
   - MSE must fall clearly.
   - Samples from 3–4 training captions must reproduce THEIR images.
   - Wrong image for caption → misaligned rows. Same image for every caption → conditioning not wired.
     Colour noise → scaling/normalization bug.
4. Measured dropout rate ≈ 15%.
5. Speed: 1K steps fp32 vs bf16 → pick precision, recompute total hours for 400K steps.

### 8. `sample_t2i.py`
- Load EMA weights, CLIP, VAE, ∅.
- `y = cat([e_prompts, e_null.expand(n, -1)])`; `z_T = randn(n, 4, 32, 32)`.
- DPM-Solver++:
  - `DPMSolverMultistepScheduler(num_train_timesteps=1000, beta_schedule="linear", beta_start=1e-4, beta_end=0.02, algorithm_type="dpmsolver++", solver_order=2, prediction_type="epsilon")`
  - `set_timesteps(20)`
  - Loop: model on `cat([z, z])`, take `[:, :4]`, CFG, `sched.step(...).prev_sample`.
- DDPM: `create_diffusion("250").p_sample_loop(model.forward_with_cfg, ..., clip_denoised=False, model_kwargs=dict(y=y, cfg_scale=s))`, then keep the first half.
- Decode: `vae.decode(z / 0.18215).sample` → `((x + 1) / 2).clamp(0, 1)`.

### 9. Evaluation
- 10K val2014 captions (one per image) → generate 10K images (record sampler, steps, s, ckpt).
- Reference: the matching 10K real images, same 256 center crop.
- FID-10K with clean-fid. Never mix FID implementations or sample counts.
- CLIP Score with `torchmetrics.multimodal.CLIPScore` (ViT-L/14, ViT-B/32).
- CFG sweep s ∈ {1.0, 1.5, 2, 3, 4, 5} → FID vs CLIP Score plot.
- Every permanent checkpoint (50K steps): FID-2K to track convergence / memorization
  (~1,240 epochs over 82,783 images).

## Rules for Claude when working in this repo
- Do not edit `diffusion/`. Keep the original DiT code structure; add new files rather than rewriting.
- Keep the adaLN-Zero zero-initialization intact.
- Any deviation from the table above must be flagged and recorded in `REPORT_NOTES.md`.
- When a hyperparameter isn't specified here, say so instead of assuming.
- Update `REPORT_NOTES.md` (changes log + results tables) whenever code behavior or results change.
