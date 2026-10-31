# DiT-S/4 Text-Conditioned Training

> Configuración efectiva del checkpoint compartido: **150.000 pasos y batch 128**,
> según sus `args` y `log.txt`. El log registra **45.174 imágenes** y ejecución CPU
> con AMP bf16. El batch 256 y la RTX 3090 mencionados abajo describen la
> configuración prevista, no la ejecución registrada. El extractor de Linda
> selecciona imágenes con personas; el informe en `results/dit_s4_150k/INFORME.md`
> explica la evaluación sobre val2014 sin ese filtro.

Desde la raíz del repositorio, el script textual publicado por David es `train.py`.
Importa las dependencias de `base_code/`; se puede comprobar su interfaz sin entrenar:

```bash
OMP_NUM_THREADS=2 PYTHONPATH=base_code .venv/bin/python train.py --help
```

`base_code/train.py` es el script original condicionado por clases. El script de
David y su condición nula de ceros se conservan para documentar el entrenamiento
recibido; no se ha relanzado ni corregido el entrenamiento al preparar la evaluación.

Se usa **AdamW lr=1e-4, weight_decay=0, batch=256, T=1000, β lineal 1e-4→0.02, EMA=0.9999** siguiendo exactamente el paper (Peebles & Xie, 2023). DiT-S/4 (depth=12, d=384, patch=4, ~33M params) es la variante más ligera del paper, escogida por el presupuesto de cómputo (RTX 3090). La pérdida híbrida MSE + 0.001·VLB con `learn_sigma=True` se mantiene igual que en el paper.

**Hiperparámetros nuevos introducidos por el condicionamiento textual (no existen en el paper original):**
- `cfg_dropout=0.15`: probabilidad de sustituir el embedding de texto por el embedding nulo ∅ durante el entrenamiento, necesario para habilitar Classifier-Free Guidance en inferencia.
- `text_dim=768`: dimensión del embedding pooled de CLIP ViT-L/14, proyectado al espacio del modelo mediante `TextEmbedder` (Linear→SiLU→Linear).
- `cfg_scale` (inferencia): escala de guía CFG; se evalúa en s∈{1.0, 1.5, 2, 3, 4, 5} — no existe en el DiT class-conditional original.

**Checkpoint final:** `results/000-DiT-S-4/checkpoints/last.pt` (step 150 000, ~504 MB) — compartir vía Drive (no versionado en git por tamaño).
