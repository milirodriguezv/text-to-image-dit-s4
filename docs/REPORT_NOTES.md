# Report Notes — Text-to-Image DiT-S/4

Source of truth for the final report, together with `reparto_trabajo_DiT_detallado.md` (who did what and
the task-by-task status). If anything else disagrees with these two files, including reports and notes on
other branches, these two prevail. If they disagree with each other, this file prevails, and within it §2.1
prevails over the planned values in §2 and §3.

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

The final run differs from this plan in data, training length, hardware and ∅. §2.1 is the reference for
what was actually used; plan values elsewhere are marked where they differ.

---

## 2. Changes relative to the original DiT repository

| Component | Original DiT | Our version | Why |
|---|---|---|---|
| Conditioning input | Class label (0–999) | CLIP `e_pooled` (768D) of a caption | Task changes from class- to text-conditional |
| Label/text embedder | `LabelEmbedder`: lookup table with 1001 entries (1000 classes + learned null) | `TextEmbedder`: MLP 768→384→384 (SiLU) + frozen ∅ | Continuous text vectors need a projection, not a lookup |
| Null condition ∅ | Learned extra class embedding | Planned: frozen CLIP embedding of `""`. Final run: all-zero vector (§2.1) | Text has a natural "empty" value; see §3.4 |
| Condition dropout | 10% | 15% | As specified in our proposal |
| Dataset | ImageNet, ImageFolder, images loaded per step | Planned: COCO train2014. Final run: 45,174 train2014 images with people (§2.1). Precomputed latents + text embeddings in both | Speed; see §3.2 |
| VAE encoding | On the fly every step (`latent_dist.sample()`) | Offline: store mean+std, sample in the Dataset | Same stochasticity at a fraction of the cost |
| Augmentation | Random horizontal flip on pixels | Both flips pre-encoded, one picked at random | Flip must happen before encoding |
| CFG in `forward_with_cfg` | Guidance applied to 3 of 4 latent channels (reproducibility quirk noted in repo) | Guidance on all 4 channels | Standard CFG formulation |
| Samplers | DDPM (250 steps), DDIM | + DPM-Solver++ (20 steps, diffusers) | Fast default sampler per proposal |
| Training loop | Epoch-based, no resume | `--max-steps`, resume, rolling checkpoints, periodic sample grids, extra logging | A long single-process run must survive interruptions (planned ~1.5 days on GPU; final run ≈21 h on CPU) |
| Precision | fp32 + TF32 | bf16 autocast on model forward (if validated) | Planned ~1.3–1.8× speed on RTX 3090. Final run: bf16 on CPU |
| Unchanged | Patchify, 2D sin-cos pos-embed, timestep embedder, DiT blocks, adaLN-Zero init, final layer, diffusion module, hybrid loss, AdamW, EMA | — | Core of the paper, reused as-is |

## 2.1 Final implementation as built (run `000-DiT-S-4`)
This section declares what the final run actually used. Where it differs from §2 and §3, this section
describes the run and those sections describe the original plan. Sources: the training log, the
committed code, the training README and David's checks on the server (31/10).

### Final training configuration
| Item | Value |
|---|---|
| Model | DiT-S/4: depth 12, hidden 384, 6 heads, patch 4, 64 tokens, `learn_sigma=True`; 33,003,776 parameters |
| Conditioning | adaLN-Zero-Text: CLIP ViT-L/14 `e_pooled` (768) through `TextEmbedder` (Linear→SiLU→Linear) |
| Text dropout | 0.15, replaced by ∅ |
| ∅ | All-zero vector (1, 768), not CLIP("") |
| Data | 45,174 train2014 images with people; person-centered crop at 256; original + flip; 5 captions per image |
| Optimizer | AdamW, lr 1e-4, weight decay 0, no warmup, no gradient clipping |
| Batch / length | 128 / 150,000 steps (≈425 epochs) |
| Diffusion | T = 1000, linear β 1e-4 → 0.02 |
| Loss | Hybrid: MSE + vb, i.e. L_simple + 0.001·L_vlb |
| EMA | 0.9999 |
| Precision / hardware | bf16 autocast on the model forward; CPU, ≈2.0 steps/s, ≈21 h |
| Saving and sampling | Rolling checkpoint and grid every 10K, permanent checkpoint every 50K; grids: EMA, DPM-Solver++ 20 steps, s = 4.0, seed 0 |
| Final logged values | Loss 0.1684, MSE 0.1614, vb 0.0071 at step 150,000 |

### ∅ = zeros vs CLIP("")
- **Cause:** `train.py` did not pass `null_path`, so `TextEmbedder` kept its default zeros.
- **Effect on the model:** for 15% of training samples the caption was replaced by zeros. The projection
  maps zeros to a constant `y_emb`, so the model learns its own "no caption" signal, similar to DiT's
  learned null class. CLIP("") would instead have used a real text embedding from CLIP.
- **Expected impact:** no large effect is expected if the same ∅ is used in training, sampling and
  evaluation, because CFG needs a consistent unconditional signal. This is an expectation only: no
  run with CLIP("") exists to compare against.
- **At sampling:** ∅ is used at every step when s > 1. At s = 1 only the conditional prediction is used.
- **Empty prompts:** a prompt `""` produces CLIP(""), not ∅, so it is not the unconditional case for this model.

### Deviations from the fixed specification
| Item | Specification | Final run | Evidence | Status |
|---|---|---|---|---|
| Training images | COCO train2014, 82,783 | 45,174: images with at least one person box and 5 captions | `log.txt`; `load_images` | Confirmed |
| Crop | ADM center crop of the full image | Square crop around the union of person boxes + 20% margin, clipped to the image, resized bicubic to 256 | `preprocess_image`; David (31/10) | Stated by David: features generated with the latest version of Linda's extractor |
| Global batch | 256 | 128 | `log.txt`; training README | Confirmed |
| Training length | 400K steps | 150K steps | `log.txt` | Confirmed |
| Hardware | RTX 3090 | CPU (AVX-512 BF16 autodetected), ≈2.0 steps/s, ≈21 h | `log.txt`; `train.py` | Confirmed |
| Null embedding ∅ | Frozen CLIP(`""`) | All-zero vector (1, 768) | David's check of the final checkpoint (31/10) | Confirmed by David |
| Data loading | `DistributedSampler`, launched with `torchrun` | Single process, `DataLoader(shuffle=True)` | `train.py` | Confirmed |
| Seen samples | ≈1,237 epochs (102.4M samples) | ≈425 epochs (19.2M samples, 18.8% of plan) | Computed from the above | Confirmed |

### New components added for the implementation
| Component | Role | Branch |
|---|---|---|
| `TextEmbedder`, CFG on all 4 ε channels | Text conditioning (§2) | `mili` |
| `text_utils.py` | CLIP loading and `encode_texts`, shared by ∅, captions and periodic sampling | `linda` |
| `make_null_embedding.py` | Generates ∅ | `linda` |
| `extract_features.py` | Person filter and crop; resumable extraction with memmaps and progress files; also writes `image_ids.npy` | `linda` |
| `createCSV.py` | Writes `captions.csv` (`image_id`, `caption`) | `linda` |
| `dataset_t2i.py` | `CocoLatentDataset`; file name differs from the planned `datasets_t2i.py` | `linda` |
| `src/train.py` | `AutocastModelWrapper` (bf16 on the model forward only), `--max-steps`, `--resume`, `last.pt` every 10K, permanent checkpoint every 50K, DPM-Solver++ grids every 10K, `--use-amp`, `--cfg-dropout` | `david` |
| `sample_t2i.py` | Text-prompt sampling: DPM-Solver++ 20 / DDPM 250, CFG; loads EMA and ∅ from the checkpoint; refuses a zero ∅ unless `--allow-zero-null` is passed | `Daniel` |
| `eval/`, `verify_evaluation.py`, `report_evaluation.py` | Frozen reference manifest, generation, FID, CLIP Score, plots, integrity checks and report export | `Daniel` |

### Software and run settings
- **Training arguments (from the checkpoint):** model DiT-S/4, image size 256, max steps 150,000, global batch
  128, seed 0, lr 1e-4, text dropout 0.15, 4 workers, logging every 100 steps, `--use-amp auto`, no resume.
- **Checkpoint:** `last.pt`, EMA weights at step 150,000, 528,048,707 bytes, SHA-256 `f693fd73…`. Strict state
  dict match, all weights finite, ∅ norm 0.
- **Evaluation versions:** torch 2.14.1, torchvision 0.29.1, timm 1.0.30, diffusers 0.35.2, transformers 4.57.6,
  torchmetrics 1.9.0, clean-fid 0.1.35. The training environment's versions are not recorded here.

### Evaluation as executed (branch `Daniel`, commit `2074491`)
| Item | Specification / proposal | Evaluation |
|---|---|---|
| Checkpoint | EMA | EMA 150K (`last.pt`, SHA-256 `f693fd73…`), ∅ = zeros kept from the checkpoint |
| Reference images | 10K val2014 | 10K val2014: IDs sorted, shuffled with seed 0, first 10K kept; **not filtered by people**; first caption of each image in COCO annotation order |
| Reference crop | 256 center crop | 256 center crop (ADM), as specified, **but not the person crop used in training** |
| Samplers and s | DPM-Solver++ 20, s ∈ {1, 1.5, 2, 3, 4, 5}; DDPM 250 at the best s | As specified; DDPM at s = 4 (lowest FID in the sweep) |
| FID-2K | Permanent checkpoints (50K, 100K, 150K) | **150K only.** No convergence curve |
| Hardware | RTX 3090 | Apple Silicon MPS, generation batch 8 |
| Seeds | Not specified | One seed; batch seed = 0 + index of the batch's first image |

Results are in §6 and the reviews are in §7 (31/10 and 01/11).

### Conditions for reporting this run
- **∅ consequences (zeros confirmed by David, 31/10; handled by the evaluation).** `train.py` never
  passes `null_path`, so `TextEmbedder` kept its default zeros. Sampling and evaluation of this run must
  take ∅ from the checkpoint (`y_embedder.null_embedding`, built with `null_path=None`) and must not load
  `null_empty_string.npy`, otherwise CFG would use a vector the model never saw as ∅. `sample_t2i.py`
  does this, and all eight evaluation traces record `zeros_from_training_checkpoint`. Fix `train.py`
  before any new run that should follow the specification; resuming this run keeps the zeros because
  the checkpoint restores the buffer.
- **Feature provenance: stated by David, not independently checked.** David reported on 31/10 that the
  features were generated with the latest version of Linda's `extract_features.py`. That version stores unscaled statistics and `dataset_t2i.py` applies 0.18215 once, which is correct.
  The note in §5.2 of `origin/linda` and a comment in `dataset_t2i.py` say the opposite.
- **Evaluation and training use different data.** The evaluation uses unfiltered val2014 center crops,
  while the model was trained on person crops. The absolute FID and CLIP Score must be reported with
  this difference. The with/without-people breakdown (§6) describes the gap but does not prove its cause.
- **Not run separately:** VAE round trip, overfit test, and the fp32 vs bf16 comparison (§5). Without the
  overfit test, a code error cannot be fully ruled out, and poor quality cannot be attributed to one cause.
- **Complementary evaluation (decision 31/10):** the real-image CLIP Score and the with/without-people
  breakdown are done and checked (§3.12, §6, §7).
- **Out of scope (decision 31/10):** the overfit test and FID-2K at 50K and 100K will not be run; they are
  listed as future improvements in §8. The report therefore cannot fully rule out a code error, attribute
  the poor quality to one cause, or make claims about convergence.
- **Resume bug:** `update_ema(ema, model, decay=0)` runs after a checkpoint is loaded and overwrites the
  EMA. This run did not resume (`log.txt`), so it is unaffected; fix it before continuing from 150K.

---

## 3. Design decisions and rationale

### 3.1 Dataset split: train2014 / val2014
- COCO train2017 contains most val2014 images.
- Training on train2014 (82,783 images) and evaluating on 10K val2014 images/captions keeps the
  evaluation set unseen and follows the usual text-to-image COCO protocol.
- Cost: ~35K fewer training images than train2017.
- **Final run:** uses only the 45,174 train2014 images with people (§2.1). val2014 remains unseen.

### 3.2 Offline feature extraction
- The VAE encoder runs at full 256×256 resolution and costs far more per image than a DiT-S/4 forward
  pass, which sees only 64 tokens.
- VAE and CLIP are frozen, so their outputs never change. Computing them once (≈1 h) instead of every
  step for ~1,240 planned epochs (≈425 in the final run) removes most of the per-step cost.
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
- **Final run deviation:** the run used an all-zero ∅ instead, because `train.py` did not pass
  `null_path` (§2.1). Replaced captions were the zero vector during training, and
  periodic sampling used the same vector, so training and sampling are consistent. The zero vector is
  not the embedding of a real text, so the benefits listed above do not apply to this run. The MLP still
  maps it to a constant `y_emb` that the model learns as "no caption".

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
- **Final run:** batch 128 and 150K steps on CPU (§2.1). The values below are the original plan, except
  that optimizer settings, EMA and no clipping were kept.
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
- This estimate was never measured on the RTX 3090. Final run: ≈2.0 steps/s on CPU, ≈21 h (§6).

### 3.10 Sampling
- Default: DPM-Solver++ (2nd-order multistep, 20 steps), configured with the training schedule
  (1000 steps, linear β 1e-4 → 0.02, ε-prediction).
- Alternative: the repo's DDPM with 250 respaced steps, using the learned Σ_θ.
- CFG: ε = ε_uncond + s·(ε_cond − ε_uncond). The conditional and unconditional branches are batched
  into one forward pass.

#### CFG explained intuitively (for the method section)
- **What the model predicts.** Sampling starts from pure noise. At every step the model answers
  "which part of this is noise, so it can be removed?". That answer is ε.
- **Two opinions, exaggerate the difference.** At every step we ask for two answers about the same
  latent:
  - with the caption ("remove the noise thinking of *a dog on the beach*") → `ε_cond`;
  - with ∅ ("remove the noise thinking of *any image*") → `ε_uncond`.

  Their difference `ε_cond − ε_uncond` is "what the caption contributes". CFG amplifies it by `s`:
  `ε = ε_uncond + s·(ε_cond − ε_uncond)`. `s = 1` is the plain conditional prediction; `s = 4` pushes the
  sample four times as far in the caption's direction (higher CLIP Score, lower diversity).
- **Doubled batch.** Both questions go through the model in a single forward pass: latents `[z, z]`,
  conditions `[e_prompt, ∅]`. `forward_with_cfg` splits the halves, applies the formula and returns
  the guided ε in both halves; callers keep `samples.chunk(2)[0]`.
- **The 3-channel quirk we fixed.** The latent (and therefore ε) has 4 channels. The original DiT code
  guides only channels 0–2 for exact reproducibility of the paper's numbers; channel 3 stays at
  `s = 1` whatever scale is requested. We apply guidance to all 4 channels (standard CFG), so the CFG
  sweep s ∈ {1, 1.5, 2, 3, 4, 5} really guides the whole latent.
- **Σ_θ is not guided.** The 4 variance channels (how much randomness each reverse step adds) are used
  as predicted, as in DiT and GLIDE.
- Verified 06/10 (commit `a304609`): with random weights the output matches the manual formula for
  s ∈ {0, 1, 4}; channel 3 is now guided; Σ is unguided; the repo's DDPM sampler runs without NaN.

### 3.11 VAE: `sd-vae-ft-ema` for encoding and decoding
- ft-EMA and ft-MSE share the same encoder; only the decoder was fine-tuned (model card; DiT paper, Appendix D).
- Model card, COCO 2017 val at 256×256: ft-EMA has rFID 4.42 and PSNR 23.8; ft-MSE has rFID 4.70 and PSNR 24.5,
  and the card describes ft-MSE outputs as somewhat smoother.
- The DiT paper reports comparable results across decoders on ImageNet 256×256 (DiT-XL/2-G, s = 1.5, FID-50K):
  ft-EMA 2.27, ft-MSE 2.30, original LDM decoder 2.46.
- We use ft-EMA throughout for consistency.

### 3.12 Evaluation protocol
**The protocol below was executed on the full val2014 distribution with center crops, while the model
trained on person crops (§2.1).** Results therefore measure generalization beyond the training domain as
well as model quality, and the two effects cannot be separated with the current numbers. The center crop
follows the specification. Sampling uses the checkpoint's ∅.

- **FID-10K:** 10K generated images (one per val2014 caption) vs the 10K corresponding real images,
  same 256 center crop, computed with **clean-fid**.
  - Numbers are not comparable to the DiT paper (different dataset/task) or across FID
    implementations and sample counts.
- **CLIP Score:** max(mean(100·cos(image, text)), 0) via torchmetrics 1.9.
  - ViT-L/14 (primary) and ViT-B/32 (secondary). ViT-L/14 is the same CLIP model family and size as our
    text encoder; ViT-B/32 is a different CLIP model, but not an independent family.
- **CFG sweep:** s ∈ {1.0, 1.5, 2, 3, 4, 5} → FID vs CLIP Score trade-off curve.
- **Qualitative:** sample grids of 8 fixed prompts with a fixed seed every 10K steps.

#### What each metric is and what it tells us
| Metric | What it is | What it tells us | What it does not tell us |
|---|---|---|---|
| **FID** (lower is better) | Passes real and generated images through Inception-v3, fits a Gaussian to each set of features and computes the Fréchet distance: `‖μ_r − μ_g‖² + Tr(Σ_r + Σ_g − 2(Σ_r Σ_g)^½)`. 0 means identical feature statistics | How close the generated set is to the real set overall, which reflects both realism and diversity | Whether an image matches its caption, since it never sees text. It is a set-level value, not a per-image score |
| **FID-10K, FID-2K, FID-N** | FID computed with 10,000, 2,000 or N images per side | The same as FID, for that sample size | Values with different N are not comparable: FID depends on sample size, as seen in 94.19 (2K) vs 71.22 (10K) for the same configuration |
| **CLIP Score** (higher is better) | `100·cos(image, text)` for each image and its caption using CLIP embeddings, averaged over all pairs; the zero floor applies to the average (`max(mean, 0)`). At DPM s = 4, no individual score is negative | How well images match their captions according to CLIP | Realism or diversity. It is not a percentage of correct images, and its practical range is well below 100 |
| **CLIP L/14 and B/32** | The same score with two CLIP models | L/14 is the primary measure. B/32 gives a second view from a different CLIP model | L/14 and B/32 values are not comparable with each other |
| **CLIP Score of real images** | CLIP Score of the real references with the same captions | A practical reference for the generated score: how much a real COCO image scores with its caption | A strict maximum: at DPM s = 4, 9.3% (L/14) and 11.7% (B/32) of generated images score above their real counterpart |
| **Metrics by group** | FID-N and CLIP Score separately for images with and without people | Whether results differ between the two groups | The cause of a difference: one group may simply be easier, so the result can be compatible with an effect of the person filter but cannot prove it |

FID and CLIP Score measure different things, so we report them together: the CFG sweep shows how s affects
both. Neither replaces looking at the samples.

#### Complementary analyses (decision 31/10; results in §6)
Both analyses reuse the existing PNGs, references and manifest; no new images are generated.

1. **CLIP Score of real images.** CLIP L/14 and B/32 on the 10K real references, each with the caption used
   to generate its counterpart.
2. **Metrics with and without people, for DPM-Solver++ 20 at s = 4.**
   - **Groups:** an image is "with people" if its ID has at least one annotation in
     `person_keypoints_val2014.json`, and "without people" otherwise.
   - **Size:** N is the size of the smaller group. Draw N images from each group with a fixed seed and save
     the ID lists.
   - **Metrics:** FID-N of generated vs real images with the same IDs, plus CLIP L/14 and B/32 of both
     generated and real images, for each group.
   - **Reading:** FID-N is compared only between these two groups, not with FID-10K. Images keep the center
     crop, not the training crop.


**Out of scope:** the overfit test, FID-2K at 50K and 100K, and a separate s = 4 evaluation with
`--preprocess person-crop`. See the future improvements in §8.

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
| Param count ≈33.4M (CLAUDE.md / task split) | Actual count is 33,003,776 | Target ≈33.0M: original DiT-S/4 with 1001×384 label table = 32,945,024 (≈32.9M, matches paper) − 384,384 (table) + 443,136 (`TextEmbedder`). Includes the frozen `pos_embed` (24,576, `requires_grad=False`); ∅ is a buffer, not counted. Corrected in CLAUDE.md and the task split (06/10) |

---

## 5. Sanity checks (record outcome and date)
| Check | Expected | Result | Date |
|---|---|---|---|
| VAE round trip of stored latent | Matches cropped original | Not run separately | |
| Param count | ≈33.0M (corrected, see §4) | 33,003,776 ✓ | 06/10 |
| Output at init | Shape (2,8,32,32), all zeros | (2,8,32,32), all zeros ✓ (adaLN-Zero intact) | 06/10 |
| Overfit 64 samples, 2K steps | MSE falls; captions reproduce their own images | Not run separately | |
| Measured text dropout | ≈15% | 0.1501 ✓ (N=100,000 via `token_drop`; repeat runs 0.15008, 0.1506) | 06/10 |
| fp32 vs bf16, 500 steps | Overlapping loss; bf16 faster | Not run separately | |

The VAE round trip, overfit test and fp32 vs bf16 comparison were not run separately. The coherence of the
samples generated during training gives indirect validation only. The overfit test is out of scope and is
listed as a future improvement (§8).

---

## 6. Experiment log
| Run ID | Date | Config changes | Steps | it/s | Wall time | Notes |
|---|---|---|---|---|---|---|
| `000-DiT-S-4` | Pushed 28/10; run date not logged | See §2.1: 45,174 person-crop images, batch 128, CPU bf16, ∅ = zeros | 150,000 | ≈2.0 | ≈21 h (from logged rates) | Final logged loss 0.1684, MSE 0.1614. Evaluated below |

### Training curve (`log.txt`, logged every 100 steps)
| Steps | Mean loss | Mean MSE | Mean vb | Mean grad norm |
|---|---|---|---|---|
| 100 (single log) | 0.5205 | 0.5084 | 0.0121 | 1.4020 |
| 100–10K | 0.1945 | 0.1868 | 0.0077 | 0.1358 |
| 10K–50K | 0.1775 | 0.1700 | 0.0075 | 0.0317 |
| 50K–100K | 0.1721 | 0.1646 | 0.0074 | 0.0229 |
| 100K–140K | 0.1697 | 0.1623 | 0.0074 | 0.0205 |
| 140K–150K | 0.1687 | 0.1614 | 0.0073 | 0.0199 |

No NaN was logged, and the largest grad norm is the first one (1.40 at step 100). The rates average 2.015
steps/s and add up to 20.9 h. The falling loss shows that training progressed, not that the model converged.

### Evaluation results
Source: `outputs/evaluation/dit_s4_150k/` (delivered in commit `2074491`). Conditions: 10K val2014 references without
a person filter, center crop, one seed, ∅ = zeros, clean-fid, torchmetrics, Apple Silicon MPS. Dani's control
reproduced the FID-10K and both CLIP Scores of DPM s = 4 from the PNGs (§7). We did not rerun generation or
FID, because the checkpoint and the full PNGs are not in git.

### Intermediate checkpoints (FID-2K, DPM-Solver++ 20 steps, s = 4.0)
| Step | FID-2K | CLIP Score (L/14) | Notes (sample grid observations) |
|---|---|---|---|
| 50K | Not evaluated | Not evaluated | Grid: baseball scene recognizable, the rest abstract or very dark |
| 100K | Not evaluated | Not evaluated | Grid: similar compositions to 50K (same seed), baseball scene clearer |
| 150K | 94.1912 | 18.3290 | CLIP B/32 24.1158. Not comparable with FID-10K. Grid: see below |

### Final evaluation (EMA, 150K)
| Sampler | Steps | s | FID-10K | CLIP Score L/14 | CLIP Score B/32 |
|---|---|---|---|---|---|
| DPM-Solver++ | 20 | 1.0 | 98.4370 | 15.3453 | 21.0660 |
| DPM-Solver++ | 20 | 1.5 | 87.0201 | 16.2527 | 22.1661 |
| DPM-Solver++ | 20 | 2.0 | 79.0465 | 16.9160 | 22.9279 |
| DPM-Solver++ | 20 | 3.0 | 71.9762 | 17.8069 | 23.7768 |
| DPM-Solver++ | 20 | 4.0 | **71.2168** | 18.2717 | 24.1091 |
| DPM-Solver++ | 20 | 5.0 | 73.2326 | **18.4661** | **24.2080** |
| DDPM | 250 | 4.0 | 76.6554 | 18.3189 | 24.1666 |

**Supported reading:** in this protocol, raising s from 1 to 4 lowers FID and raises both CLIP Scores. At
s = 5, CLIP Score still rises and FID worsens. At s = 4, DPM-Solver++ 20 has a lower FID than DDPM 250,
and their CLIP Scores are close. With one seed, the 0.76 FID gap between s = 3 and s = 4 should not be
presented as a clear win.

### Complementary analyses (EMA 150K, DPM-Solver++ 20, s = 4)
Source: `outputs/evaluation/dit_s4_150k/analisis_complementario/` (delivered in commit `f7de252`). No new images
were generated. Images keep the center crop and ∅ stays the checkpoint's zeros.

**CLIP Score on the same 10,000 image-caption pairs**
| Images | CLIP Score L/14 | CLIP Score B/32 |
|---|---|---|
| Real | 25.5643 | 30.1501 |
| Generated | 18.2717 | 24.1091 |

**Balanced groups by person annotations (seed 0)**
| Group | N | FID-4750 | CLIP L/14 generated | CLIP L/14 real | CLIP B/32 generated | CLIP B/32 real |
|---|---|---|---|---|---|---|
| With person annotations | 4,750 | 56.3083 | 19.7638 | 25.5768 | 25.8457 | 30.0862 |
| Without person annotations | 4,750 | 116.3012 | 16.6117 | 25.5185 | 22.1943 | 30.2021 |

Of the 10K references, 5,250 have person annotations and 4,750 do not, so the second group is used whole and
the first is subsampled.

**Supported reading:** real images score similarly in both groups, but generated images score higher and have
a lower FID-4750 in the group with people. The result is compatible with the training set being person crops,
but does not prove it: the groups also differ in content, captions and composition. Real-image CLIP is a
reference, not a ceiling or a percentage of quality. FID-4750 is compared only between these two groups, not
with FID-10K or FID-2K.

**Uncertainty estimate (computed by us):** bootstrap over images (5,000 resamples, 95% interval)
from `scores_por_imagen.csv`, for this one generated sample.
| Difference | CLIP L/14 | CLIP B/32 |
|---|---|---|
| Generated, with minus without people | 3.15 [3.02, 3.29] | 3.65 [3.52, 3.79] |
| Real, with minus without people | 0.06 [-0.09, 0.21] | -0.12 [-0.24, 0.02] |
| Real-generated gap, without minus with people | 3.09 [2.88, 3.31] | 3.77 [3.59, 3.95] |

Each group's images are resampled independently. No interval exists for FID-4750, because the Inception
features are not committed. The intervals cover the choice of images only, not a new seed or a retrained model.

### Repeated images (hashes in `run.json`)
| Run | All-black images | With / without people | Other identical pairs |
|---|---|---|---|
| DPM s ≤ 3 | 0 | – | None |
| DPM s = 4 | 3 | 0 / 3 | IDs 8762, 205863 (both without people) |
| DPM s = 5 | 3 | 1 / 2 | None |
| DDPM s = 4 | 7 | 0 / 7 | IDs 164469, 219269 (both with people) |

The all-black images appear only at s ≥ 4, and 12 of the 13 are in the group without people. The rate is at
most 0.07%, too low to explain the overall FID. The cause is unknown, and the identical pairs were not
inspected.

### Qualitative observations
Periodic grids (`outputs/training/samples/`) use EMA, DPM-Solver++ 20, s = 4, seed 0. In order, the prompts are:
"A dog sitting on a beach at sunset", "A red car parked in front of a building", "A woman riding a bicycle in a
park", "A plate of food on a wooden table", "A cat sleeping on a couch", "A group of people playing baseball",
"An airplane flying over mountains" and "A vase of flowers next to a window".

Grid at 150K, as seen by us:
| Prompt | Observation |
|---|---|
| Baseball | Field, players and stands recognizable; players deformed. The clearest result |
| Woman on bicycle | Outdoor park scene with a deformed figure; no clear bicycle |
| Food | Warm food-like colors and textures on a surface; no clear plate or table |
| Dog on beach | Sunset colors only; no dog |
| Airplane | Blue sky colors with a blurred blob; no clear airplane |
| Cat | Dark purple textured forms; no clear cat |
| Car and vase | Mostly dark; subject not recognizable |

Across the 50K, 100K and 150K grids, compositions stay similar and details change. This is the same seed on 8
prompts and is not a measure of convergence.

Eight evaluation examples (the first eight in the manifest, not hand-picked), DPM and DDPM at s = 4, as described
by Dani and consistent with our 31/10 check:
- A person walking into the ocean gives a recognizable beach, with a deformed figure and object.
- A bird among purple flowers gives colored regions and no clearly identifiable bird.
- A woman next to a giraffe gives deformed people or faces, and the animal is not clearly resolved.
- Two giraffes running give mostly vegetation; neither the two giraffes nor the action can be confirmed.
- A man on a horse watching another wrestle a cow gives animals or a herd, without that relation.

**Reading:** results match categories and scenes in part, but fail on anatomy, object count and complex actions.
Some high-s images look blocky and oversaturated. These are examples, not a global measure, and do not show that
one sampler looks better than the other.

---

## 7. Observations, issues and deviations
_(Date each entry. Include bugs found, anything that deviated from the plan, and why.)_

- **06/10 — `models.py` implemented (Persona 2, branch `mili`).** Commits: `76118fb` (TextEmbedder
  interface stub), `f65bf53` (TextEmbedder logic), `869ec46` (proj init normal std 0.02 + default
  `class_dropout_prob` 0.1 → 0.15), `a304609` (CFG on all 4 ε channels), `457e9fe` (tests).
- **06/10 — Interface decisions (not specified in CLAUDE.md; ours):**
  - Constructor: `DiT_models["DiT-S/4"](input_size=32, text_dim=768, class_dropout_prob=0.15, null_path=...)`.
  - `null_path` is optional. ∅ is a persistent buffer `y_embedder.null_embedding`, saved in the
    checkpoint, so it is only needed when creating the model for training; when loading a
    checkpoint pass `null_path=None`.
  - Accepted ∅ file: `.npy` of shape (1,768) or (768,), cast to float32; wrong size → `AssertionError`.
  - ∅ itself is produced by Persona 1 (`null_empty_string.npy`); `models.py` does not compute it
    (tests use a synthetic vector).
  - Name `class_dropout_prob` kept for compatibility, although it is now text dropout.
  - Default changed 0.1 → 0.15 so that forgetting to pass it does not silently train with 10%.
- **06/10 — Recommendation for `train.py` (Persona 3):** add
  `assert model.y_embedder.null_embedding.abs().sum() > 0` after building the model; if `null_path`
  is forgotten, ∅ silently stays all zeros.
- **06/10 — adaLN-Zero gradient propagation (expected, not a bug):** step 0: only
  `final_layer.linear` (W, b) gets a non-zero gradient; step 1: 30 tensors (final adaLN, the 12
  blocks' adaLN, patch embed via residual paths, final linear); step 2: all 134, incl. `TextEmbedder`.
  Zero gradient on the text embedder during the first 2 steps is therefore normal.
- **06/10 — Test suite** (`base_code/tests/test_models.py`): run
  `.venv/bin/python -m pytest base_code/tests -q` → 19 passed, 1 skipped (CUDA bf16), ~6 s. bf16
  autocast on CPU works. Comparisons use `allclose(atol=1e-5)` because batched vs per-row float32
  matmuls differ by ~1e-7.
- **06/10 — Local test environment:** `.venv` with Python 3.13, torch 2.14.1, timm 1.0.30; CPU/MPS
  (Mac). Reminder: torch ≥ 2.6 needs `torch.load(..., weights_only=False)` for our checkpoints.
- **31/10 — ∅ = zeros in the final run.** David confirmed from the final checkpoint that ∅ is all zeros.
  `train.py` did not pass `null_path`, and the assert recommended on 06/10 was not added. The deviation is
  declared in §2.1.
- **31/10 — Review of the final run.** Deviations and reporting conditions are in §2.1. This is the only
  completed run, and the final report is built on it. The grid at 150K is uneven: the baseball
  prompt is recognizable, but several prompts without people are not, which may reflect the person-only
  training data (not tested). Repeating to specification is a larger job than the 21 h run: ≈56 h for
  400K steps at batch 128, and roughly 112 h at batch 256 if cost per step scales linearly, plus
  re-extraction of all 82,783 images.
- **31/10 — Review of the evaluation (branch `Daniel`, commit `2074491`).**
  - **Scope of the review:** code and committed traces. The checkpoint and the full PNGs were not
    available, so generation and metrics were not rerun or recomputed.
  - **Code checks with no error found:** EMA loaded with a strict state dict; ∅ taken from the checkpoint;
    CFG on 4 ε channels with `uncond + s·(cond − uncond)`; DPM-Solver++ with original-scale timesteps;
    DDPM through `create_diffusion` and `forward_with_cfg` with `clip_denoised=False`; decoding with
    `/ 0.18215`; FID restricted to the matching references, including the 2K prefix; CLIP Score paired
    image by image with its caption. `diffusion/` is unchanged, and `models.py`, `train.py`,
    `dataset_t2i.py` and `extract_features.py` match the team branches.
  - **Text encoding:** evaluation passes `attention_mask` and training does not. A test with a small,
    randomly initialized CLIP gave the same `pooler_output` (maximum difference 0.0), as padding comes
    after [EOS]. This was not repeated with the real CLIP weights.
  - **Trace checks that passed:** all 8 runs have aligned image IDs and captions and the expected seeds,
    and their configurations differ only in the expected fields. The CSV matches `metrics.json` and all 64
    committed sample PNGs match their hashes. The FID-2K images are identical to the first 2,000 images
    of DPM s = 4, even though the code fingerprint differs.
  - **Tests:** 45 passed, 1 skipped (CUDA) in a clean Python 3.11 environment with
    `requirements-eval.txt`. Our `.venv` lacks `diffusers`, so 6 tests fail there for that reason only.
  - **All-black images:** a solid-black 256×256 PNG matches the repeated hash `a3684197d1` in the runs listed
    in §6.
  - **Visual check:** the 8 committed examples show scenes and colors related to the caption, but
    objects and people are deformed, and some high-s images look blocky and oversaturated. Eight examples
    are not a global measure.
  - **Known limits at that date:** no person/non-person breakdown and no real-image CLIP Score (both added on
    01/11), no FID-2K at 50K/100K, and no confidence interval. `verify_evaluation.py` relies on absolute paths stored in the traces, so it
    only works where those paths exist.
  - **Branch history:** `Daniel` is not a merge of the team branches, and `main` does not contain the
    implementation. The results and code cited here must be read from the branches named in §10.
  - **Decision:** add only the real-image CLIP Score and the with/without-people breakdown (§3.12). The overfit
    test, FID-2K at 50K and 100K, and the `person-crop` evaluation are out of scope and listed in §8.
- **01/11 — Review of the complementary analyses (branch `Daniel`, commit `f7de252`).**
  - **Scope of the change:** the commit only adds files and edits documents; the earlier code and traces are
    unchanged.
  - **Recomputed from committed files:** all group and global CLIP means from `scores_por_imagen.csv`, the ID
    selection (`random.Random(0)`), the group partition, and row alignment with `run.json`. All match.
    File hashes also match, including the analysis code and the original run.
  - **Tests:** 50 passed, 1 skipped (CUDA), on a clean copy of `base_code/` with Python 3.11.
  - **Read, not re-run:** FID-4750 and the CLIP values per image need the full PNGs, which only Dani has. The
    code looks correct: it uses the same IDs on both sides, and its own control reproduces the original
    results (FID-10K within 0.000013, CLIP within 0.000004, per Dani's committed output).
  - **Group labels not checked against the official file:** `person_keypoints_val2014.json` is not on this
    machine. As an indirect check, a person-related word list matches 75.3% of captions with annotations and
    3.1% without.
  - **Repeated images:** crossed with the groups for all runs (§6). There are too few to draw a conclusion.
  - **Uncertainty:** the CLIP intervals in §6 are ours, computed from Dani's per-image scores.
- **01/11 — Sources for the final report.** The team will write it from this file and
  `reparto_trabajo_DiT_detallado.md` only. Dani's `INFORME.md` is not a source, so its wording is not corrected.
- **01/11 — Branches integrated into `main`.** This supersedes the earlier note that `main` lacks the implementation.
  - **Method:** `david`, `linda` and `mili` merged on top of `Daniel`, keeping each branch's history. No code file
    changed in the merges, and `diffusion/` is untouched.
  - **Conflicts:** `REPORT_NOTES.md` keeps `mili`'s version; `README_training.md` and `.gitignore` keep `Daniel`'s.
  - **Removed from the tree:** `base_code/prueba.py` and `preprocessing_test.png`. Both remain in git history.
  - **`README.md` rewritten:** it now points to the two sources of truth and documents reproduction end to end.
  - **Known problems left unfixed**, so the code remains the one that produced this run: ∅ in zeros, the EMA
    overwrite on resume (`train.py:244`) and the incorrect comment at `dataset_t2i.py:82`.
  - **Checked:** tests with Python 3.11 in a clean environment (50 passed, 1 skipped, CUDA) and `--help` of the
    main scripts. Training, generation and metrics were not re-run. The public Drive link was seen to serve
    `last.pt`, but the file was not downloaded, so its SHA-256 was not checked here.
- **01/11 — Repository reorganized (requested by Milagros).** Paths in entries before this one are the old ones.
  - **Layout:** `dit_base/` (paper code; `diffusion/` unchanged), `src/` (new code), `tests/`, `outputs/`, `docs/`.
    This deviates from the rule in `CLAUDE.md` of keeping the original structure; the rule was updated.
  - **Deleted (remain in git history):** the original README, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `visuals/`,
    `run_DiT.ipynb`, `environment.yml`, the class-conditional `train.py`, `sample.py` and `sample_ddp.py`, and the
    unread `null_empty_string.npy`. `LICENSE.txt` is kept, as the original license requires.
  - **Code edits, paths only:** `eval/run.py` (`code_fingerprint`), `eval/build_colab.py` and the tests. Model,
    training, sampling and metrics behavior is unchanged.
  - **Consequence for the code fingerprint:** new runs record a different `code_sha256` from the delivered runs,
    so a partial generation from the old code cannot be continued with `--resume`. Delivered results stay valid
    but were produced by the earlier layout (`main` at `e8bf78b`).
  - **Results folder not edited:** `outputs/evaluation/dit_s4_150k/` keeps its old paths in `INFORME.md` and traces.
  - **Audit of the README commands (real small COCO subset: 16 train, 32 val, 8 generated images):** features,
    20 training steps, checkpoint download, generation and metrics ran without errors. The checkpoint SHA-256
    matched. Not run: the `train2014.zip` download, full `execute_plan`, the analyses, anything at full scale.

---

## 8. Limitations and possible extensions (for discussion section)
- Global conditioning limits attribute binding and spatial relations; Variant B (cross-attention on
  `E_seq`) is the natural extension.
- DiT-S/4 is the smallest DiT configuration. The paper shows FID improves strongly with model size
  and smaller patch size (more tokens, more Gflops).
- 256×256 resolution limits detail. The final run used 45,174 images with people, which limits
  diversity further and may weaken prompts without people.
- The final run saw 19.2M samples, 18.8% of the planned 102.4M.
- ∅ = zeros was not compared with CLIP("") (§2.1).
- Evaluation covers one checkpoint and one seed, on data with a different distribution and crop from
  training. It does not show that the person filter causes the failures, nor how much of the FID it explains.
- **Supported claims:** on this checkpoint, set and seed, s = 4 has the lowest FID-10K in the sweep; CLIP Score
  rises with s up to 5; DPM-Solver++ 20 has a lower FID-10K than DDPM 250 at s = 4; real images score higher on
  CLIP than generated ones; and results are better in the group with people.
- **Unsupported claims:** that the person filter causes the failures; that the model converged; that a code error
  is ruled out; that s = 4 clearly beats s = 3; that 20-step DPM-Solver++ is 12.5× faster (speed was not measured);
  or any comparison with the DiT paper's FID values.
- Future improvements left out of scope:
  - **Overfit test** (64 images, 1 caption each, no flip, dropout 0, batch 64, ~2K steps), to help separate
    a code error from limited training. Until it is run, neither can be ruled out.
  - **FID-2K at 50K and 100K,** to see how results evolved during training. Without it, the report makes no
    claim about convergence.
  - **Evaluation with `person-crop`** at s = 4, reported separately, to measure the model on the training crop.
- Possible ablations: MLP vs Linear text projection; learned vs CLIP("") ∅; guidance scale;
  sampler steps.

## 9. References to cite
- Peebles & Xie (2023), *Scalable Diffusion Models with Transformers* (DiT), arXiv:2212.09748.
- Ho, Jain & Abbeel (2020), *Denoising Diffusion Probabilistic Models*, arXiv:2006.11239.
- Nichol & Dhariwal (2021), *Improved Denoising Diffusion Probabilistic Models* (learned variance, hybrid loss), arXiv:2102.09672.
- Ho & Salimans (2022), *Classifier-Free Diffusion Guidance*, arXiv:2207.12598.
- Rombach et al. (2022), *High-Resolution Image Synthesis with Latent Diffusion Models*, arXiv:2112.10752.
- Stability AI, `sd-vae-ft-ema` model card, huggingface.co/stabilityai/sd-vae-ft-ema (rFID and PSNR in §3.11).
- Radford et al. (2021), *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), arXiv:2103.00020.
- Lu et al. (2022), *DPM-Solver++*, arXiv:2211.01095.
- Lin et al. (2014), *Microsoft COCO: Common Objects in Context*, arXiv:1405.0312.
- Heusel et al. (2017), *GANs Trained by a Two Time-Scale Update Rule* (FID), arXiv:1706.08500.
- Parmar et al. (2022), *On Aliased Resizing and Surprising Subtleties in GAN Evaluation* (clean-fid), arXiv:2104.11222.
- Hessel et al. (2021), *CLIPScore*, arXiv:2104.08718.
- Esser et al. (2024), *Scaling Rectified Flow Transformers* (SD3, pooled-text MLP precedent), arXiv:2403.03206.

## 10. Where the report's material lives
Everything versioned is on `main`.

| Material | Location |
|---|---|
| Sweep figures | `outputs/evaluation/dit_s4_150k/figuras/cfg_metrics.png` and `fid_vs_clip.png` |
| 8 evaluation examples (real, all s, DDPM) | `outputs/evaluation/dit_s4_150k/muestras/` |
| Training grids every 10K | `outputs/training/samples/` |
| Training log | `outputs/training/log.txt` |
| Numeric traces and per-image scores | `outputs/evaluation/dit_s4_150k/datos/` and `analisis_complementario/` |
| Code | `dit_base/` (paper code) and `src/` (new code); see `README.md` |
| Checkpoint | `last.pt`, not in git; public Drive link in `README.md` and `src/eval/fetch_checkpoint.py` |
| Full 10K PNG sets | Not in git; held by Dani. Can be regenerated with `eval.execute_plan` |
