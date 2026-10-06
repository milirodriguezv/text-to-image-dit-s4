# Reparto del trabajo — DiT-S/4 Text-to-Image

## Objetivo del reparto

Dividir el trabajo entre 4 personas de forma equilibrada, maximizando el trabajo en paralelo y reduciendo al mínimo las dependencias entre personas.

La idea es que **las cuatro personas puedan empezar a trabajar desde el principio**, utilizando datos/modelos dummy cuando todavía no estén disponibles los módulos de los demás.

Las dependencias inevitables son:

```text
Datos + Modelo
      ↓
  Entrenamiento
      ↓
  Checkpoint
      ↓
Sampling + Evaluación
```

---

# Persona 1 — Datos, extracción de features y Dataset

## Objetivo

Dejar completamente preparado el pipeline que convierte las imágenes y captions de COCO en los datos que necesita el modelo.

## 1. Preparación de los datos

- Descargar/preparar:
  - `train2014`
  - `val2014`
  - `captions_train2014.json`
  - `captions_val2014.json`
- Comprobar que las imágenes y captions están correctamente asociadas.
- Comprobar que se utilizan las primeras 5 captions de cada imagen.
- No utilizar `train2017`.

## 2. Implementar `extract_features.py`

Implementar todo el proceso offline.

### Imágenes

```text
Imagen
  ↓
Center crop 256×256
  ↓
Normalización [-1,1]
  ↓
VAE
  ↓
mean + std
```

Para cada imagen:

- Procesar la imagen original.
- Procesar también el horizontal flip.
- Utilizar el VAE congelado.
- Obtener `mean` y `std`.
- Guardar `cat([mean, std])` en fp16.

### Texto

```text
Caption
  ↓
CLIP tokenizer
  ↓
CLIP
  ↓
e_pooled (768)
```

Para cada imagen:

- Tomar las primeras 5 captions.
- Tokenizarlas con CLIP.
- Obtener `pooler_output`.
- Guardarlo en fp16.

### Archivos generados

```text
latents.npy
e_pooled.npy
null_empty_string.npy
```

Para validation:

- Guardar los embeddings necesarios para las 10K imágenes/captions de evaluación.
- Guardar la correspondencia entre imagen y caption.

## 3. Implementar `datasets_t2i.py`

Utilizar:

```python
np.load(..., mmap_mode="r")
```

El Dataset debe:

- Elegir aleatoriamente el flip.
- Elegir aleatoriamente una de las 5 captions.
- Construir `z_0`.
- Devolver los datos en `float32`.

La salida debe ser:

```python
z0, e_pooled = dataset[i]
```

con:

```text
z0       → (4, 32, 32)
e_pooled → (768,)
```

## 4. Tests

Comprobar:

- Dimensiones de los features.
- Correspondencia entre las filas de los arrays.
- Al menos 5 captions por imagen.
- VAE round-trip.
- Que la imagen reconstruida tiene sentido visualmente.
- Que el scaling `0.18215` se aplica correctamente.

## Puede trabajar independientemente

No necesita código de Persona 2 ni Persona 3 para comenzar.

Puede desarrollar y probar todo el pipeline desde el principio.

## Entregables

```text
extract_features.py
datasets_t2i.py
features/
```

---

# Persona 2 — Arquitectura DiT + Text Conditioning + CFG

## Objetivo

Transformar el DiT original en el modelo text-conditioned especificado en el proyecto.

## 1. Implementar `TextEmbedder`

Sustituir:

```text
LabelEmbedder
```

por:

```text
TextEmbedder
```

con la arquitectura:

```text
768 → 384 → 384
       SiLU
```

Implementar:

- Proyección del embedding de CLIP.
- Embedding nulo `∅`.
- `register_buffer`.
- Text dropout del 15 %.
- `force_drop_ids`.

## 2. Adaptar `DiT`

Modificar:

```python
DiT.__init__()
```

para utilizar:

```text
text_dim = 768
```

Mantener:

- DiT-S/4.
- adaLN-Zero.
- Inicialización original.
- `learn_sigma=True`.

## 3. Adaptar `forward`

Comprobar que el conditioning funciona como:

```text
c = t_emb + y_emb
```

donde `y_emb` procede del texto.

## 4. Adaptar `forward_with_cfg`

Implementar correctamente el conditioning:

```text
prompt embedding
       +
null embedding
       ↓
       CFG
```

y utilizar los **4 canales de ruido** para CFG.

## 5. Tests

Comprobar:

```text
≈33.0M parámetros (33.003.776)
```

Entrada:

```text
(B, 4, 32, 32)
```

Salida:

```text
(B, 8, 32, 32)
```

Además:

- Zero-output inicial.
- Text dropout ≈15 %.
- Conditional/unconditional.
- CFG.
- Embedding nulo.

## Puede trabajar independientemente

No necesita el Dataset real de Persona 1.

Puede utilizar embeddings ficticios:

```python
torch.randn(B, 768)
```

para probar el modelo.

## Entregable

```text
models.py
```

---

# Persona 3 — Entrenamiento + Integración

## Objetivo

Construir todo el pipeline de entrenamiento y dejarlo preparado para conectar el Dataset y el modelo definitivos.

**Importante:** no debe esperar a que Personas 1 y 2 terminen para empezar.

## 1. Preparar `train.py`

Mientras los demás trabajan, implementar:

- DataLoader.
- DistributedSampler.
- Optimizer.
- EMA.
- Checkpoints.
- Logging.
- Resume.
- Número máximo de pasos.
- Grad norm.
- Steps/sec.
- Guardado periódico.

## 2. Implementar la loss

Integrar:

```text
L = L_simple + 0.001 · L_vlb
```

y registrar:

```text
loss
mse
vb
```

## 3. Precision

Implementar:

```text
fp32
bf16
```

El autocast debe estar **solo alrededor del forward del modelo**, no alrededor de todo `training_losses`.

## 4. Checkpoints

Implementar:

```text
last.pt          → cada 10K pasos
{step:07d}.pt   → cada 50K pasos
```

Además:

```text
--resume
--max-steps
```

## 5. Sampling durante el entrenamiento

Añadir:

- 8 prompts fijos.
- Seed 0.
- Modelo EMA.
- DPM-Solver++.
- 20 pasos.
- CFG = 4.
- Una muestra cada 10K pasos.

## 6. Trabajar inicialmente con datos/modelo dummy

Antes de recibir los módulos definitivos, probar el pipeline utilizando datos artificiales:

```python
z0 = torch.randn(...)
y = torch.randn(...)
```

y un modelo compatible.

El objetivo es comprobar que el pipeline de entrenamiento funciona antes de integrar el código real.

## 7. Integración final

Cuando Personas 1 y 2 terminen:

- Sustituir el Dataset dummy por `CocoLatentDataset`.
- Sustituir el modelo dummy por el DiT-S/4 definitivo.
- Comprobar que las interfaces son compatibles.

## 8. Sanity checks finales

Realizar:

- Overfit con 64 imágenes.
- Aproximadamente 2K pasos.
- Comprobar que el MSE disminuye.
- Comprobar que el conditioning funciona.
- Comprobar que un caption produce la imagen correspondiente.
- Comprobar el dropout.
- Comparar fp32/bf16.
- Medir steps/sec.
- Estimar la duración de las 400K iteraciones.

## Dependencias

Puede desarrollar prácticamente todo de forma independiente.

Solo necesita a Personas 1 y 2 para la **integración final**.

## Entregables

```text
train.py
checkpoints/
samples/
```

---

# Persona 4 — Sampling + Evaluación + Métricas

## Objetivo

Construir todo el pipeline que transforma:

```text
checkpoint + prompt
        ↓
      imagen
        ↓
     métricas
```

**Importante:** tampoco debe esperar al entrenamiento real para empezar.

---

# 1. Implementar `sample_t2i.py`

## Carga

Implementar la carga de:

- EMA.
- CLIP.
- VAE.
- Embedding nulo.

## DPM-Solver++

Configurar:

```text
1000 diffusion steps
20 sampling steps
order = 2
prediction = epsilon
```

## DDPM

Implementar:

```text
250 pasos
```

## CFG

Implementar el sweep:

```text
s ∈ {1, 1.5, 2, 3, 4, 5}
```

## Decoding

Implementar:

```text
latent
  ↓
/ 0.18215
  ↓
VAE decode
  ↓
imagen [0,1]
```

---

# 2. Implementar la evaluación

Crear:

```text
eval/
```

## FID

Implementar:

```text
FID-10K
```

utilizando `clean-fid`.

## CLIP Score

Calcular:

```text
ViT-L/14
ViT-B/32
```

## CFG sweep

Generar resultados para:

```text
s = 1
s = 1.5
s = 2
s = 3
s = 4
s = 5
```

Preparar los resultados para la gráfica:

```text
FID vs CLIP Score
```

## FID-2K

Preparar el pipeline para evaluar los checkpoints de:

```text
50K
100K
150K
...
400K
```

## 3. Trabajar con checkpoint dummy

No tiene que esperar al entrenamiento de Persona 3.

Puede crear/utilizar un checkpoint artificial para comprobar:

- Carga del modelo.
- Generación de imágenes.
- DPM-Solver++.
- DDPM.
- CFG.
- Guardado de imágenes.
- Pipeline de métricas.
- Generación de gráficas.

Cuando llegue el checkpoint real, únicamente se cambia la ruta/configuración necesaria.

## Entregables

```text
sample_t2i.py
eval/
results/
figuras/
```

---

# 🔄 Organización del trabajo en paralelo

## Fase 1 — Desarrollo independiente

Las cuatro personas empiezan simultáneamente:

| Persona | Trabajo |
|---|---|
| **1** | Features + Dataset |
| **2** | DiT + Text Conditioning |
| **3** | Training usando mocks |
| **4** | Sampling + Evaluación usando mocks |

## Fase 2 — Tests individuales

| Persona | Tests |
|---|---|
| **1** | Dataset, features, VAE round-trip |
| **2** | Modelo, shapes, parámetros, dropout, CFG |
| **3** | Loss, optimizer, EMA, checkpoints, fp32/bf16 |
| **4** | Sampling, CFG, métricas y gráficas |

## Fase 3 — Integración

Primera sincronización:

```text
Persona 1 ──► Dataset ──┐
                        │
                        ▼
                     Persona 3
                        ▲
                        │
Persona 2 ──► Modelo ───┘
```

Persona 3 integra Dataset + Modelo y realiza los sanity checks de entrenamiento.

## Fase 4 — Entrenamiento real

```text
Persona 3
    │
    ▼
400K steps
    │
    ▼
Checkpoints
```

Mientras tanto, las Personas 1, 2 y 4 pueden seguir trabajando en:

- validación,
- documentación,
- pruebas,
- análisis de resultados,
- preparación del informe.

## Fase 5 — Evaluación

Segunda sincronización:

```text
Persona 3 ──► Checkpoint
                  │
                  ▼
              Persona 4
                  │
                  ▼
         Sampling + Evaluation
```

---

# 🔗 Interfaces que deben acordarse antes de empezar

Para evitar conflictos, estas interfaces deberían mantenerse fijas.

## Dataset → Training

```python
z0, e_pooled = dataset[i]
```

con:

```text
z0       = (4, 32, 32)
e_pooled = (768,)
```

## Modelo → Training

```python
output = model(x, t, y)
```

con:

```text
x      = (B, 4, 32, 32)
t      = (B,)
y      = (B, 768)
output = (B, 8, 32, 32)
```

## Training → Sampling

El checkpoint debe contener:

```text
checkpoint["model"]
checkpoint["ema"]
checkpoint["opt"]
checkpoint["args"]
```

---

# 📁 Archivos asignados

| Archivo | Responsable |
|---|---|
| `extract_features.py` | Persona 1 |
| `datasets_t2i.py` | Persona 1 |
| `models.py` | Persona 2 |
| `train.py` | Persona 3 |
| `sample_t2i.py` | Persona 4 |
| `eval/` | Persona 4 |
| `diffusion/` | **No modificar** |

---

# ⚠️ Regla importante

Cada persona debería trabajar principalmente sobre sus propios archivos.

No modificar `diffusion/`, ya que el proyecto especifica que debe mantenerse intacto.

Las tres interfaces críticas son:

```text
Dataset → Training
Modelo → Training
Checkpoint → Sampling/Evaluación
```

Si estas interfaces se mantienen estables, las cuatro personas pueden trabajar de forma prácticamente independiente y la integración final será mucho más sencilla.
