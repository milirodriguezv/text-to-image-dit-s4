# Report Notes — Text-to-Image DiT-S/4

Running documentation for writing the final report: what changed from the original DiT, what we
decided and why, corrections to our own proposal docs, and places to log results.

---

## 1. Summary of the approach
We adapt the Diffusion Transformer (DiT; Peebles & Xie, 2023) from class-conditional ImageNet
generation to text-conditioned generation on MS-COCO Captions. The smallest configuration, DiT-S/4,
operates in the latent space of a frozen Stable Diffusion VAE (256×256×3 → 4×32×32). Captions are
encoded by a frozen CLIP ViT-L/14 text encoder.

Conditioning follows **adaLN-Zero-Text (global modulation)**:
- The pooled caption vector `e_pooled` is projected to the model width and added to the timestep
  embedding.
- Each transformer block regresses its scale, shift and gate parameters from this sum, exactly like
  the class label in the original DiT.

---

## 2. Changes relative to the original DiT repository

| Component | Original DiT | Our version | Why |
|---|---|---|---|
| Conditioning input | Class label (0–999) | CLIP `e_pooled` (768D) of a caption | Task changes from class- to text-conditional |
| Label/text embedder | `LabelEmbedder`: lookup table with 1001 entries (1000 classes + learned null) | `TextEmbedder`: MLP 768→384→384 (SiLU) + frozen ∅ | Continuous text vectors need a projection, not a lookup |
| Null condition ∅ | Learned extra class embedding | Frozen CLIP embedding of `""` | Text has a natural "empty" value; see §3.4 |
| Condition dropout | 10% | 15% | As specified in our proposal |
| Dataset | ImageNet, ImageFolder, images loaded per step | COCO train2014, precomputed latents + text embeddings | Speed; see §3.2 |
| VAE encoding | On the fly every step (`latent_dist.sample()`) | Offline: store mean+std, sample in the Dataset | Same stochasticity at a fraction of the cost |
| Augmentation | Random horizontal flip on pixels | Both flips pre-encoded, one picked at random | Flip must happen before encoding |
| CFG in `forward_with_cfg` | Guidance applied to 3 of 4 latent channels (reproducibility quirk noted in repo) | Guidance on all 4 channels | Standard CFG formulation |
| Samplers | DDPM (250 steps), DDIM | + DPM-Solver++ (20 steps, diffusers) | Fast default sampler per proposal |
| Training loop | Epoch-based, no resume | `--max-steps`, resume, rolling checkpoints, periodic sample grids, extra logging | Single-GPU run of ~1.5 days must survive interruptions |
| Precision | fp32 + TF32 | bf16 autocast on model forward (if validated) | ~1.3–1.8× speed on RTX 3090 |
| Unchanged | Patchify, 2D sin-cos pos-embed, timestep embedder, DiT blocks, adaLN-Zero init, final layer, diffusion module, hybrid loss, AdamW, EMA | — | Core of the paper, reused as-is |

---

## 3. Design decisions and rationale

### 3.1 Dataset split: train2014 / val2014
- COCO train2017 contains most val2014 images.
- Training on train2014 (82,783 images) and evaluating on 10K val2014 images/captions keeps the
  evaluation set unseen and follows the usual text-to-image COCO protocol.
- Cost: ~35K fewer training images than train2017.

### 3.2 Offline feature extraction
- The VAE encoder runs at full 256×256 resolution and costs far more per image than a DiT-S/4 forward
  pass, which sees only 64 tokens.
- VAE and CLIP are frozen, so their outputs never change. Computing them once (≈1 h) instead of every
  step for ~1,240 epochs removes most of the per-step cost.
- We store the VAE's posterior mean and std rather than a single sample. A fresh latent is drawn every
  time an image is used, reproducing the original repo's behaviour.
- `E_seq` is not stored (~50 GB), since only Variant B uses it.

### 3.3 Text encoder: frozen CLIP ViT-L/14, pooled [EOS] output
- Variant A compresses the caption into one vector.
- CLIP's pooled vector is trained contrastively to summarize what an image matching the caption
  contains, which makes it the natural single-vector representation.
- Encoders without a trained pooled output (e.g. T5) are better suited to cross-attention (Variant B).
- COCO captions (~10–15 words) are far below CLIP's 77-token limit.
- Same encoder as Stable Diffusion 1.x; SD3 also uses a pooled CLIP vector for global conditioning.

### 3.4 Null embedding ∅ = CLIP("")
- Classifier-free guidance requires the model to also denoise without a caption, so 15% of training
  captions are replaced by ∅.
- The original DiT learns a "null class" because ImageNet's 1000 classes have no natural empty value.
- Text does have one, the empty caption, so we use its frozen CLIP embedding. It:
  - lies in the same space as real caption embeddings;
  - adds no learned parameters;
  - is the standard choice in text-to-image models (Stable Diffusion).

### 3.5 Text projection: 2-layer MLP (768 → 384 → 384, SiLU)
- The projection is needed because `e_pooled` (768) must match the model width (d=384) before it is
  summed with `t_emb`.
- A trainable layer also translates CLIP's matching-oriented space into one useful for steering
  denoising.
- We use the same Linear–SiLU–Linear structure as the existing timestep embedder, so both conditioning
  signals are processed symmetrically. SD3 uses the same design for its pooled text vector.
- Cost: ~0.44M parameters (≈1.3% of the model); ~0.15M more than a single Linear.
- Possible ablation: MLP vs Linear.

### 3.6 How the caption acts inside the network (for the method section)
Tokens: 64 × 384. Conditioning vector: `c = t_emb + y_emb` (384).

Each of the 12 blocks maps `c` through SiLU + Linear to six 384-dimensional vectors
(β1, γ1, α1, β2, γ2, α2), and applies:

- `x = x + α1 · Attn(LN(x)·(1+γ1) + β1)`
- `x = x + α2 · MLP(LN(x)·(1+γ2) + β2)`

Properties of this mechanism:
- The same γ, β, α apply to all 64 tokens. The caption scales and shifts feature channels globally;
  it is never attended to.
- Spatial layout emerges from self-attention among image tokens. The caption biases which features
  are emphasized, and because scaling is multiplicative, it amplifies existing evidence rather than
  placing content.
- α is zero-initialized, so every block starts as the identity (adaLN-Zero) and training is stable
  from step 0.
- Expected limitation: weaker attribute binding and spatial relations (e.g. "a red cube left of a
  blue ball") than cross-attention (Variant B).

### 3.7 Loss: hybrid, learned variance (`learn_sigma=True`)
- The model outputs 8 channels: ε_θ (noise to remove) and Σ_θ (how much randomness each reverse step
  adds).
- `loss = MSE(ε_θ, ε) + vb`:
  - MSE trains ε_θ only.
  - vb, the KL between the true reverse-step Gaussian and the model's, trains Σ_θ only (stop-gradient
    on the mean).
  - The single-timestep vb estimate equals L_vlb / 1000, so the loss is L_simple + 0.001·L_vlb
    (Nichol & Dhariwal's L_hybrid), exactly as in DiT.
- A learned Σ_θ is what makes 250-step DDPM work well. DPM-Solver++ is deterministic and ignores Σ_θ.

### 3.8 Training configuration
- AdamW, lr 1e-4, weight decay 0, global batch 256, no warmup, EMA 0.9999 — as in DiT.
- **400K steps**, matching the DiT paper's standard comparison point.
  - That is ≈1,240 epochs over train2014.
  - Intermediate FID-2K every 50K steps monitors convergence and possible memorization.
- **No gradient clipping** (as in paper), but the gradient norm is logged so instability would be
  visible.
- **bf16 autocast** on the model forward if a 500-step comparison with fp32 shows matching loss
  curves. Weights, optimizer and EMA stay fp32; the loss is computed in fp32.

### 3.9 Compute estimate
- FLOPs ≈ 6·N·D with N ≈ 33M parameters and D = 400K × 256 × 64 ≈ 6.6·10⁹ tokens,
  giving ≈ 1.3·10¹⁸ FLOPs (≈1.3 ExaFLOPs).
- At a realistic ~10 TFLOPS on an RTX 3090: ≈0.3 s/step → ≈33–37 h.
- Consistent with the proposal's 9–38 h range.
- Measured value: see §6.

### 3.10 Sampling
- Default: DPM-Solver++ (2nd-order multistep, 20 steps), configured with the training schedule
  (1000 steps, linear β 1e-4 → 0.02, ε-prediction).
- Alternative: the repo's DDPM with 250 respaced steps, using the learned Σ_θ.
- CFG: ε = ε_uncond + s·(ε_cond − ε_uncond). The conditional and unconditional branches are batched
  into one forward pass.

### 3.11 VAE: `sd-vae-ft-ema` for encoding and decoding
- ft-EMA and ft-MSE share the same encoder; only the decoder was fine-tuned.
- ft-EMA gives slightly sharper reconstructions (better rFID); ft-MSE is smoother (higher PSNR).
- We use ft-EMA throughout for consistency.
- TODO: verify and cite the DiT appendix comparison of decoders.

### 3.12 Evaluation protocol
- **FID-10K:** 10K generated images (one per val2014 caption) vs the 10K corresponding real images,
  same 256 center crop, computed with **clean-fid**.
  - Numbers are not comparable to the DiT paper (different dataset/task) or across FID
    implementations and sample counts.
- **CLIP Score:** 100·max(cos(image, text), 0) via torchmetrics.
  - ViT-L/14 (primary) and ViT-B/32 (secondary, independent of our conditioning encoder family).
- **CFG sweep:** s ∈ {1.0, 1.5, 2, 3, 4, 5} → FID vs CLIP Score trade-off curve.
- **Qualitative:** sample grids of 8 fixed prompts with a fixed seed every 10K steps.

---

## 4. Corrections to our proposal documents
| Doc statement | Issue | Resolution |
|---|---|---|
| Loss is MSE only (L_simple) while the model outputs Σ_θ | MSE gives Σ_θ no gradient, so it would remain untrained | Hybrid loss L_simple + 0.001·L_vlb (repo/paper default) |
| Encode with `sd-vae-ft-mse`, decode with "ft-EMA" | Inconsistent (though compatible: same encoder) | `sd-vae-ft-ema` for both |
| t ∈ [1, 1000] | Code uses t ∈ {0, …, 999} | Same thing, zero-indexed; note in the report |
| CFG scale "s = 1.5 to 4.0" vs "s ∈ [1.0, 5.0]" | Two ranges quoted | Sweep {1.0, 1.5, 2, 3, 4, 5} |
| Extract `E_seq` in Step 1 | Unused by Variant A; ~50 GB | Not stored; regenerate only if Variant B is explored |
| Dataset "MS-COCO Captions" (split unspecified) | train2017 overlaps val2014 | train2014 / val2014 |

---

## 5. Sanity checks (record outcome and date)
| Check | Expected | Result | Date |
|---|---|---|---|
| VAE round trip of stored latent | Matches cropped original | | |
| Param count | ≈33.4M | | |
| Output at init | Shape (2,8,32,32), all zeros | | |
| Overfit 64 samples, 2K steps | MSE falls; captions reproduce their own images | | |
| Measured text dropout | ≈15% | | |
| fp32 vs bf16, 500 steps | Overlapping loss; bf16 faster | | |

---

## 5.1 Completed implementation: CLIP text utilities and null embedding
The repository already includes the first text-conditioning utilities needed for the text-to-image variant:

- [base_code/text_utils.py](base_code/text_utils.py) loads the frozen CLIP tokenizer and text model from Hugging Face, freezes the model parameters, and exposes a helper to encode one or more captions using CLIP's pooled output.
- The tokenizer is configured with padding to length 77 and truncation enabled, so captions are normalized to the same shape expected by the model.
- [base_code/make_null_embedding.py](base_code/make_null_embedding.py) builds the null conditioning vector by encoding the empty string, converts it to NumPy float16, and saves it to [features/null_empty_string.npy](features/null_empty_string.npy).
- This produces the empty-text embedding used for classifier-free guidance, allowing the model to learn both conditional and unconditional denoising behavior.
- The data flow is consistent with the project specification: the CLIP vector is pooled into a 768-D embedding and is the same representation that will later be projected into the DiT model width.

---

## 6. Experiment log
| Run ID | Date | Config changes | Steps | it/s | Wall time | Notes |
|---|---|---|---|---|---|---|
| | | | | | | |

### Intermediate checkpoints (FID-2K, DPM-Solver++ 20 steps, s = 4.0)
| Step | FID-2K | CLIP Score (L/14) | Notes (sample grid observations) |
|---|---|---|---|
| 50K | | | |
| 100K | | | |
| 150K | | | |
| 200K | | | |
| 250K | | | |
| 300K | | | |
| 350K | | | |
| 400K | | | |

### Final evaluation (EMA, 400K)
| Sampler | Steps | s | FID-10K | CLIP Score L/14 | CLIP Score B/32 |
|---|---|---|---|---|---|
| DPM-Solver++ | 20 | 1.0 | | | |
| DPM-Solver++ | 20 | 1.5 | | | |
| DPM-Solver++ | 20 | 2.0 | | | |
| DPM-Solver++ | 20 | 3.0 | | | |
| DPM-Solver++ | 20 | 4.0 | | | |
| DPM-Solver++ | 20 | 5.0 | | | |
| DDPM | 250 | best s | | | |

---

## 7. Observations, issues and deviations
_(Date each entry. Include bugs found, anything that deviated from the plan, and why.)_

-

---

## 8. Limitations and possible extensions (for discussion section)
- Global conditioning limits attribute binding and spatial relations; Variant B (cross-attention on
  `E_seq`) is the natural extension.
- DiT-S/4 is the smallest DiT configuration. The paper shows FID improves strongly with model size
  and smaller patch size (more tokens, more Gflops).
- 256×256 resolution and COCO's ~83K images limit diversity and detail.
- Possible ablations: MLP vs Linear text projection; learned vs CLIP("") ∅; guidance scale;
  sampler steps.

## 9. References to cite
- Peebles & Xie (2023), *Scalable Diffusion Models with Transformers* (DiT).
- Ho, Jain & Abbeel (2020), *DDPM*.
- Nichol & Dhariwal (2021), *Improved DDPM* (learned variance, hybrid loss).
- Ho & Salimans (2022), *Classifier-Free Diffusion Guidance*.
- Rombach et al. (2022), *Latent Diffusion Models* (VAE latent space).
- Radford et al. (2021), *CLIP*.
- Lu et al. (2022), *DPM-Solver++*.
- Lin et al. (2014), *Microsoft COCO*.
- Heusel et al. (2017), *FID*; Parmar et al. (2022), *clean-fid*; Hessel et al. (2021), *CLIPScore*.
- Esser et al. (2024), *SD3* (pooled-text MLP precedent).
