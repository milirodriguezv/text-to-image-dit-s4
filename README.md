# DiT-S/4 condicionado por texto — MS-COCO

Adaptación de DiT (Peebles & Xie, 2023) para generar imágenes de 256×256 a partir de
texto, con condicionamiento adaLN-Zero-Text sobre latentes VAE y embeddings CLIP.
La licencia del DiT original está en `dit_base/LICENSE.txt`.

**Fuentes de verdad:** [`docs/REPORT_NOTES.md`](docs/REPORT_NOTES.md) y
[`docs/reparto_trabajo_DiT_detallado.md`](docs/reparto_trabajo_DiT_detallado.md). Si
difieren, manda `REPORT_NOTES.md`, y dentro de él §2.1 manda sobre el plan de §2 y §3.
Ningún otro documento del repo es fuente del informe final.

## Resultados (EMA, 150.000 pasos)

Entrenamiento con 45.174 imágenes de train2014 con personas, recortadas alrededor de
ellas, y batch 128. Evaluación con 10.000 imágenes de val2014 sin filtrar por personas,
con recorte central y una sola semilla. Esa diferencia limita la lectura de los valores
absolutos. El detalle y las limitaciones están en `REPORT_NOTES.md` (§2.1, §6, §7 y §8).

| Sampler | s | FID-10K | CLIP L/14 | CLIP B/32 |
|---|---:|---:|---:|---:|
| DPM-Solver++ 20 | 1 | 98,44 | 15,35 | 21,07 |
| DPM-Solver++ 20 | 1,5 | 87,02 | 16,25 | 22,17 |
| DPM-Solver++ 20 | 2 | 79,05 | 16,92 | 22,93 |
| DPM-Solver++ 20 | 3 | 71,98 | 17,81 | 23,78 |
| DPM-Solver++ 20 | 4 | 71,22 | 18,27 | 24,11 |
| DPM-Solver++ 20 | 5 | 73,23 | 18,47 | 24,21 |
| DDPM 250 | 4 | 76,66 | 18,32 | 24,17 |

CLIP de las imágenes reales: 25,56 (L/14) y 30,15 (B/32).

## Organización

```text
dit_base/   Código del paper: diffusion/ (sin cambios), models.py y download.py (adaptados), LICENSE.txt
src/        Código nuevo: features, dataset, entrenamiento, sampling y evaluación (eval/)
tests/      Tests
outputs/    training/ (log y grillas cada 10K) y evaluation/dit_s4_150k/ (métricas, trazas, figuras, ejemplos)
docs/       Notas del informe, reparto y guías de entrenamiento y evaluación
```

| Archivo en `src/` | Función |
|---|---|
| `extract_features.py`, `createCSV.py`, `make_null_embedding.py`, `text_utils.py` | Features VAE y CLIP |
| `dataset_t2i.py` | Dataset sobre los features |
| `train.py` | Entrenamiento |
| `sample_t2i.py` | Generación desde texto |
| `eval/`, `analyze_evaluation.py`, `report_evaluation.py`, `verify_evaluation.py` | Evaluación y análisis |

Los archivos de `outputs/evaluation/dit_s4_150k/` se conservan como se entregaron, así
que sus trazas e `INFORME.md` todavía muestran las rutas anteriores a esta organización.

## Qué no está en git

| Material | Cómo obtenerlo |
|---|---|
| COCO | Paso 2 |
| `features/` | Paso 3 |
| Checkpoint `last.pt` (528.048.707 bytes) | `eval.fetch_checkpoint` lo baja de [Drive](https://drive.google.com/file/d/1j-Y6_RWIFsyos7pAT5eBBhAcPtEjJ95a/view) y verifica el SHA-256 `f693fd7391979856e7059994643e755239276ba5d66a31becf1ebf1e18e93c6b` |
| PNG completos de la evaluación | Paso 6 |
| CLIP ViT-L/14 y VAE `sd-vae-ft-ema` | Se bajan solos de Hugging Face la primera vez |
| Corridas nuevas | Quedan en `outputs/runs/`, que git ignora |

## Reproducción

Todos los comandos se ejecutan desde la raíz del repo. Para consultar los resultados no
hace falta correr nada. Los pasos 2 a 6 completos son largos: el entrenamiento registrado
tardó unas 21 h y la evaluación genera 72.000 imágenes.

### 1. Entorno y tests

Con Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/dit_base:$PWD/src"
OMP_NUM_THREADS=2 python -m pytest tests -q
```

Resultado esperado: 50 tests pasados y 1 omitido si no hay CUDA. El `export` hace falta
en cada terminal nueva. Estas son las versiones de la evaluación. Las del entrenamiento
no quedaron registradas, así que este entorno no garantiza un resultado idéntico.

### 2. Datos

```bash
mkdir -p coco
curl -L -o coco/train2014.zip http://images.cocodataset.org/zips/train2014.zip
unzip -q coco/train2014.zip -d coco
python -m eval.fetch_coco --root coco --count 10000 --seed 0
unzip -qn coco/annotations_trainval2014.zip -d coco
```

`train2014.zip` pesa unos 13 GB. `eval.fetch_coco` baja las anotaciones y las 10.000
imágenes de val2014 de la evaluación. El último `unzip` agrega `person_keypoints_*`, que
la extracción necesita. No entrenar con train2017: contiene la mayor parte de val2014.

### 3. Features

```bash
python src/extract_features.py
python src/createCSV.py
python src/make_null_embedding.py
```

Leen `coco/` y escriben en `features/train/`. Solo procesan imágenes con personas y 5
captions. `extract_features.py` usa siempre CPU (`device = "cpu"`) y retoma lo que quedó
a medias. `make_null_embedding.py` genera `null_empty_string.npy`, pero **`train.py` no
lo usa**. Según David, los features de la corrida salieron de esta versión del extractor;
no se verificó de forma independiente.

### 4. Entrenamiento

```bash
python src/train.py --feature-path features/train --results-dir outputs/runs \
  --max-steps 150000 --global-batch-size 128 --global-seed 0 \
  --lr 1e-4 --cfg-dropout 0.15 --use-amp auto
```

Son los argumentos guardados en el checkpoint, salvo `--results-dir`. El resultado queda
en `outputs/runs/NNN-DiT-S-4/`: `log.txt`, `checkpoints/` y `samples/`.

- **Dispositivo:** usa CUDA o CPU; no usa MPS.
- **Precisión:** `--use-amp auto` activa bf16 con CUDA o en CPU con AVX-512 BF16. Sin eso
  corre en fp32. La corrida original fue en CPU con bf16.

Más detalle en [`docs/README_training.md`](docs/README_training.md).

### 5. Generar imágenes

```bash
python -m eval.fetch_checkpoint --output checkpoints/david/last.pt
python src/sample_t2i.py --ckpt checkpoints/david/last.pt \
  --prompt "A dog on a beach" --output outputs/runs/ejemplo \
  --sampler dpm --steps 20 --cfg-scale 4 --seed 0 --batch-size 8 \
  --allow-zero-null --device cpu
```

`--output` debe ser una carpeta que no exista. `--device` admite `cpu`, `mps`, `cuda` o `auto`.

### 6. Evaluar

```bash
python -m eval.prepare --captions coco/annotations/captions_val2014.json \
  --images-dir coco/val2014 --output outputs/runs/references_val2014_10k_seed0 \
  --count 10000 --seed 0 --preprocess center-crop

OMP_NUM_THREADS=2 python -u -m eval.execute_plan --ckpt checkpoints/david/last.pt \
  --manifest outputs/runs/references_val2014_10k_seed0/manifest.json \
  --output outputs/runs/reproduccion_150k \
  --generation-device cuda --metric-device cuda --clip-device cuda \
  --batch-size 8 --metric-batch-size 32 --allow-zero-null
```

`execute_plan` genera y calcula las métricas del barrido completo. Para una sola
configuración se usan `eval.run generate` y `eval.run metrics`. Esos comandos, los análisis
complementarios (`src/analyze_evaluation.py`) y la reanudación están en
[`docs/eval_README.md`](docs/eval_README.md).

Los análisis necesitan los PNG completos. `src/verify_evaluation.py` también, y además
depende de las rutas absolutas guardadas en las trazas, así que solo funciona donde esas
rutas existen.

## Problemas conocidos

Se dejaron sin corregir para que el código siga siendo el que produjo esta corrida. El
detalle está en `docs/REPORT_NOTES.md` (§2.1 y §7).

- **∅ en ceros:** `src/train.py` no le pasa `null_path` al modelo, así que ∅ queda en ceros
  y no es CLIP(""), como pedía la especificación. Para reproducir esta corrida hay que
  mantenerlo; para seguir la especificación hay que corregirlo.
- **`--allow-zero-null`:** `sample_t2i.py` rechaza un ∅ en ceros sin esta opción, y evaluar
  esta corrida la necesita.
- **Reanudar entrenamiento:** `update_ema(ema, model, decay=0)` (`src/train.py:244`) corre
  después de cargar el checkpoint y pisa el EMA. No afectó a esta corrida, porque no se
  reanudó. Hay que corregirlo antes de usar `--resume`.
- **Comentario incorrecto:** `src/dataset_t2i.py:82` dice que los features ya incluyen
  0,18215. La escala se aplica una sola vez, en la línea 84.
- **Huella de código:** como esta organización movió archivos, las corridas nuevas registran
  un `code_sha256` distinto del de los resultados entregados, y una generación parcial
  hecha con el código anterior no se puede continuar con `--resume`.
