# DiT-S/4 Text-Conditioned Training

Se usa **AdamW lr=1e-4, weight_decay=0, batch=256, T=1000, β lineal 1e-4→0.02, EMA=0.9999** siguiendo exactamente el paper (Peebles & Xie, 2023). DiT-S/4 (depth=12, d=384, patch=4, ~33M params) es la variante más ligera del paper, escogida por el presupuesto de cómputo (RTX 3090). La pérdida híbrida MSE + 0.001·VLB con `learn_sigma=True` se mantiene igual que en el paper.

**Hiperparámetros nuevos introducidos por el condicionamiento textual (no existen en el paper original):**
- `cfg_dropout=0.15`: probabilidad de sustituir el embedding de texto por el embedding nulo ∅ durante el entrenamiento, necesario para habilitar Classifier-Free Guidance en inferencia.
- `text_dim=768`: dimensión del embedding pooled de CLIP ViT-L/14, proyectado al espacio del modelo mediante `TextEmbedder` (Linear→SiLU→Linear).
- `cfg_scale` (inferencia): escala de guía CFG; se evalúa en s∈{1.0, 1.5, 2, 3, 4, 5} — no existe en el DiT class-conditional original.

**Checkpoint final:** `results/000-DiT-S-4/checkpoints/last.pt` (step 150 000, ~504 MB) — compartir vía Drive (no versionado en git por tamaño).
