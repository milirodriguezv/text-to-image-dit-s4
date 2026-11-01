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

# 📌 Estado actual (01/11)

Este archivo y `REPORT_NOTES.md` son las fuentes de verdad del informe final. Si difieren, manda `REPORT_NOTES.md`. Los informes y notas de otras ramas no se usan como fuente.

| Persona | Parte | Estado |
|---|---|---|
| **1 — Linda** | Features de entrenamiento + Dataset | ✅ Terminado. La evaluación de Dani no necesitó features de validación precalculados |
| **2 — Milagros** | `models.py` + tests | ✅ Terminado |
| **3 — David** | `train.py` + entrenamiento | ✅ Terminado: 150K pasos |
| **4 — Dani** | Sampling + evaluación | ✅ Entregado (commits `2074491` y `f7de252`): evaluación de 150K y análisis complementarios |

## Qué cambió respecto al plan original

| Ítem | Plan | Lo que se hizo |
|---|---|---|
| Imágenes | 82.783 de `train2014` | 45.174 de `train2014`, con personas y al menos 5 captions |
| Recorte | Center crop 256×256 | Cuadrado alrededor de las personas, con 20 % de margen, llevado a 256×256 |
| ∅ | CLIP(`""`) | Vector de ceros |
| Batch | 256 | 128 |
| Pasos | 400K | 150K (≈425 épocas) |
| Hardware | RTX 3090 | CPU, ≈2 pasos/s, ≈21 h |
| Ejecución | `torchrun` + `DistributedSampler` | Un solo proceso + `shuffle=True` |
| Evaluación: imágenes | 10K de `val2014` | 10K de `val2014` sin filtrar por personas, con center crop. Es distinto del recorte de entrenamiento |
| FID-2K | Checkpoints de 50K, 100K y 150K | Solo 150K |

El detalle y las evidencias están en `REPORT_NOTES.md` (§2.1). En las secciones de abajo, **"Implementado"** indica lo que difiere del plan original.

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
  - `person_keypoints_train2014.json` (añadido: se usa para el filtro y el recorte de personas)
- Comprobar que las imágenes y captions están correctamente asociadas.
- Comprobar que se utilizan las primeras 5 captions de cada imagen.
- No utilizar `train2017`.

## 2. Implementar `extract_features.py`

Implementar todo el proceso offline.

### Imágenes

```text
Imagen con personas
  ↓
Recorte alrededor de las personas → 256×256
  ↓
Normalización [-1,1]
  ↓
VAE
  ↓
mean + std
```

**Implementado:** en lugar del center crop previsto, se usa el recuadro que abarca a todas las personas, con 20 % de margen, hecho cuadrado y limitado a los bordes de la imagen. Cerca de un borde, el recorte puede deformarse al pasar a 256×256.

Para cada imagen:

- Procesar la imagen original.
- Procesar también el horizontal flip.
- Utilizar el VAE congelado.
- Obtener `mean` y `std`.
- Guardar `cat([mean, std])` en fp16, sin escalar. `0.18215` se aplica una sola vez, en el Dataset.

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
image_ids.npy
captions.csv            (lo genera createCSV.py)
null_empty_string.npy   (lo genera make_null_embedding.py)
```

**Implementado:** `null_empty_string.npy` existe, pero el entrenamiento final no lo usó (∅ quedó en ceros).

Para validation:

- Guardar los embeddings necesarios para las imágenes/captions de evaluación.
- Guardar la correspondencia entre imagen y caption.

**Implementado:** no hizo falta extraerlos aparte. `eval/` codifica cada caption al generar y guarda la correspondencia en su `manifest.json`.

## 3. Implementar `dataset_t2i.py`

(Se planeó como `datasets_t2i.py`; el nombre final es `dataset_t2i.py`.)

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
- Que el scaling `0.18215` se aplica correctamente.

**Implementado:** se registró una prueba de extracción con 2 imágenes (dimensiones, tipos e IDs). El VAE round-trip y la revisión visual de la reconstrucción no se ejecutaron por separado; las muestras del entrenamiento solo los validan de forma indirecta.

## Puede trabajar independientemente

No necesita código de Persona 2 ni Persona 3 para comenzar.

Puede desarrollar y probar todo el pipeline desde el principio.

## Entregables

```text
extract_features.py
dataset_t2i.py
text_utils.py
make_null_embedding.py
createCSV.py
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

**Implementado:** ∅ se carga con `null_path`. Si no se pasa, queda en ceros, que es lo que ocurrió en el entrenamiento final. ∅ se guarda en el checkpoint.

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

**Resultado (06/10):** 19 tests pasan y 1 se omite porque requiere GPU. 33.003.776 parámetros; dropout medido 0,1501.

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
- Optimizer.
- EMA.
- Checkpoints.
- Logging.
- Resume.
- Número máximo de pasos.
- Grad norm.
- Steps/sec.
- Guardado periódico.

**Implementado:** un solo proceso con `DataLoader(shuffle=True)`, sin `DistributedSampler`. `train.py` está en la raíz del repo y se ejecuta con `PYTHONPATH=base_code`. No pasa `null_path`, por lo que ∅ queda en ceros. Si se reanuda el entrenamiento, antes hay que corregir `update_ema(ema, model, decay=0)`, que pisa el EMA cargado.

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

**Implementado:** el entrenamiento final usó bf16 en CPU.

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

| Chequeo | Estado |
|---|---|
| Overfit con 64 imágenes (≈2K pasos, MSE, caption → imagen) | No se ejecutó por separado |
| Dropout | ✅ 0,1501 (tests de Persona 2) |
| Comparar fp32/bf16 | No se ejecutó por separado |
| Medir steps/sec | ✅ ≈2 pasos/s en CPU (log) |
| Estimar la duración | ✅ Medida: ≈21 h para 150K pasos |

El overfit queda fuera de alcance y figura como posible mejora futura (ver Persona 4).

## Dependencias

Puede desarrollar prácticamente todo de forma independiente.

Solo necesita a Personas 1 y 2 para la **integración final**.

## Entregables

```text
src/train.py
docs/README_training.md
outputs/training/log.txt
outputs/training/samples/
last.pt   (paso 150K, en Drive, no en git)
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

## Implementado (commits `2074491` y `f7de252`)

**Entregado:** `sample_t2i.py`, `eval/`, `verify_evaluation.py`, `report_evaluation.py`, `analyze_evaluation.py`, tests y `results/dit_s4_150k/`. Se usó el checkpoint EMA de 150K, ∅ en ceros tomado del checkpoint y 10K imágenes de `val2014` sin filtrar por personas, con center crop.

| Sampler | s | FID-10K | CLIP L/14 | CLIP B/32 |
|---|---|---|---|---|
| DPM-Solver++ 20 | 1 | 98,44 | 15,35 | 21,07 |
| DPM-Solver++ 20 | 1,5 | 87,02 | 16,25 | 22,17 |
| DPM-Solver++ 20 | 2 | 79,05 | 16,92 | 22,93 |
| DPM-Solver++ 20 | 3 | 71,98 | 17,81 | 23,78 |
| DPM-Solver++ 20 | 4 | **71,22** | 18,27 | 24,11 |
| DPM-Solver++ 20 | 5 | 73,23 | **18,47** | **24,21** |
| DDPM 250 | 4 | 76,66 | 18,32 | 24,17 |

FID-2K a 150K (DPM-Solver++ 20, s = 4): 94,19. No se compara con FID-10K.

**Revisiones del 31/10 y el 01/11:** no se encontraron errores en el código ni en las trazas commiteadas. Los tests dan 50 pasados y 1 omitido. Recalculamos las medias de CLIP y la selección de grupos desde los archivos commiteados, y coinciden. No regeneramos imágenes ni recalculamos el FID, porque el checkpoint y los PNG completos no están en git; el control de Dani reproduce el FID-10K y los CLIP de s = 4. El detalle está en `REPORT_NOTES.md` (§7).

### Qué mide cada métrica

| Métrica | Qué es | Qué nos dice | Qué no nos dice |
|---|---|---|---|
| **FID** (menor es mejor) | Pasa las imágenes reales y las generadas por Inception-v3 y mide la distancia entre las estadísticas de ambos conjuntos. 0 significa estadísticas idénticas | Qué tan parecido es el conjunto generado al real, en realismo y diversidad | Si cada imagen coincide con su caption, porque no usa el texto. Es un valor del conjunto, no de cada imagen |
| **FID-10K, FID-2K, FID-N** | FID con 10.000, 2.000 o N imágenes por lado | Lo mismo que el FID, para ese tamaño | No se comparan entre sí: la misma configuración da 94,19 con 2K y 71,22 con 10K |
| **CLIP Score** (mayor es mejor) | `100·coseno(imagen, texto)` para cada imagen y su caption, promediado sobre todos los pares; el piso de cero se aplica al promedio (`max(media, 0)`) | Cuánto coinciden las imágenes con sus captions según CLIP | Realismo ni diversidad. No es un porcentaje de aciertos |
| **CLIP L/14 y B/32** | El mismo score con dos modelos de CLIP | L/14 es la medida principal y B/32 aporta una segunda mirada | Los valores de L/14 y B/32 no se comparan entre sí |
| **CLIP de imágenes reales** | CLIP Score de las imágenes reales con las mismas captions | Una referencia para interpretar el score de las generadas | Un máximo estricto: en DPM s = 4, el 9,3 % (L/14) y el 11,7 % (B/32) de las generadas superan a su imagen real |
| **Métricas por grupo** | FID-N y CLIP Score por separado, con y sin personas | Si los resultados cambian entre ambos grupos | La causa: un grupo puede ser más fácil. Puede ser compatible con un efecto del filtro de personas, pero no lo prueba |

### Análisis complementarios (decididos el 31/10, a cargo de Dani)

Solo se hacen estos dos análisis. Ninguno genera imágenes nuevas. Están entregados en el commit `f7de252`, y la lectura y los intervalos están en §6 de `REPORT_NOTES.md`:

- [x] **CLIP Score de las imágenes reales:** L/14 y B/32 sobre las 10.000 referencias, con las mismas captions usadas para generar.
- [x] **FID y CLIP Score con y sin personas (DPM-Solver++ 20, s = 4):**
  - Una imagen va a "con personas" si su ID tiene al menos una anotación en `person_keypoints_val2014.json`, y a "sin personas" si no tiene ninguna.
  - N es el tamaño del grupo más chico. Se sacan N imágenes de cada grupo con semilla fija y se guardan los IDs.
  - Por grupo se calcula el FID-N de generadas contra reales con los mismos IDs, y CLIP L/14 y B/32 de generadas y de reales.
  - El FID-N se compara solo entre estos dos grupos, no con el FID-10K. Las imágenes siguen con recorte central.

| Imágenes (10K) | CLIP L/14 | CLIP B/32 |
|---|---|---|
| Reales | 25,56 | 30,15 |
| Generadas (DPM-Solver++ 20, s = 4) | 18,27 | 24,11 |

| Grupo (N = 4.750) | FID-N | CLIP L/14 generadas | CLIP L/14 reales | CLIP B/32 generadas | CLIP B/32 reales |
|---|---|---|---|---|---|
| Con personas | 56,31 | 19,76 | 25,58 | 25,85 | 30,09 |
| Sin personas | 116,30 | 16,61 | 25,52 | 22,19 | 30,20 |

Los resultados son compatibles con haber entrenado con recortes de personas, pero no lo prueban, porque los grupos también difieren en contenido y captions.

Imágenes totalmente negras: 3 en DPM s = 4, 3 en DPM s = 5 y 7 en DDPM s = 4, con causa desconocida. 12 de las 13 están en el grupo sin personas.

### Fuera de alcance (posibles mejoras futuras para el informe)

- Prueba de sobreajuste, para ayudar a separar un error de código de la falta de entrenamiento.
- FID-2K de los checkpoints de 50K y 100K, para ver cómo evolucionaron los resultados.
- Una corrida con s = 4 y `--preprocess person-crop`, informada aparte.

---

# 1. Implementar `sample_t2i.py`

## Carga

Implementar la carga de:

- EMA.
- CLIP.
- VAE.
- Embedding nulo: **tomarlo del checkpoint** (`y_embedder.null_embedding`, creando el modelo con `null_path=None`). No usar `null_empty_string.npy`, porque el entrenamiento final usó ceros.

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

**Implementado:** se usaron 10K imágenes de `val2014` sin filtrar por personas, con center crop. El modelo entrenó con recortes de personas, así que esta diferencia debe informarse junto con el resultado.

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
```

**Implementado:** el pipeline admite otros checkpoints, pero solo se evaluó el de 150K. Por eso no hay curva de convergencia.

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
analyze_evaluation.py
outputs/evaluation/dit_s4_150k/  (incluye figuras/, muestras/ y analisis_complementario/)
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
| **1** | Dataset y features (VAE round-trip no ejecutado por separado) |
| **2** | Modelo, shapes, parámetros, dropout, CFG |
| **3** | Loss, optimizer, EMA, checkpoints (fp32/bf16 no comparado) |
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

Persona 3 integra Dataset + Modelo. Los sanity checks de overfit y fp32/bf16 no se ejecutaron por separado.

## Fase 4 — Entrenamiento real

```text
Persona 3
    │
    ▼
150K steps (✅ terminado)
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

✅ Evaluación de 150K y análisis complementarios entregados. Las ramas están integradas en `main`; `REPORT_NOTES.md` (§10) indica dónde está cada material.

---

# 🔗 Interfaces

Estas son las interfaces usadas en la implementación final.

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
output = model(x, t, y=e_pooled)
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
checkpoint["step"]
checkpoint["args"]
```

∅ está dentro de `checkpoint["ema"]`, en `y_embedder.null_embedding`, y en esta corrida vale cero. Para cargarlo con torch ≥ 2.6 hace falta `weights_only=False`.

---

# 📁 Archivos asignados

| Archivo | Responsable |
|---|---|
| `extract_features.py` | Persona 1 |
| `dataset_t2i.py` | Persona 1 |
| `text_utils.py` | Persona 1 |
| `make_null_embedding.py` | Persona 1 |
| `createCSV.py` | Persona 1 |
| `models.py` | Persona 2 |
| `train.py` | Persona 3 |
| `sample_t2i.py` | Persona 4 |
| `eval/` | Persona 4 |
| `analyze_evaluation.py` | Persona 4 |
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
