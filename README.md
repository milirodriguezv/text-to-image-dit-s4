# DiT-S/4 condicionado por texto — MS-COCO

Adaptación de DiT (Peebles & Xie, 2023) para generar imágenes de 256×256 a partir de
texto, con condicionamiento adaLN-Zero-Text sobre latentes VAE y embeddings CLIP.
La licencia del DiT original está en `base_code/LICENSE.txt`.

**Fuentes de verdad:** [`REPORT_NOTES.md`](REPORT_NOTES.md) y
[`reparto_trabajo_DiT_detallado.md`](reparto_trabajo_DiT_detallado.md). Si difieren,
manda `REPORT_NOTES.md`, y dentro de él §2.1 manda sobre el plan de §2 y §3. Ningún
otro documento del repo es fuente del informe final.

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

| Ruta | Contenido |
|---|---|
| `train.py` | Entrenamiento con texto. Se ejecuta desde la raíz |
| `base_code/models.py`, `base_code/tests/` | Modelo con `TextEmbedder` y tests |
| `base_code/extract_features.py`, `createCSV.py`, `make_null_embedding.py`, `dataset_t2i.py`, `text_utils.py` | Features y dataset |
| `base_code/sample_t2i.py`, `base_code/eval/`, `base_code/analyze_evaluation.py` | Sampling, evaluación y análisis |
| `base_code/diffusion/` | Difusión del repo original, sin cambios |
| `log.txt`, `samples/` | Log y grillas del entrenamiento cada 10K pasos |
| `results/dit_s4_150k/` | Métricas, trazas, figuras y ejemplos de la evaluación |

`base_code/train.py`, `sample.py` y `sample_ddp.py` son los scripts originales por
clases: quedan solo como referencia y no sirven para este modelo.

## Qué no está en git

| Material | Cómo obtenerlo |
|---|---|
| COCO | Paso 2 |
| `features/` | Se regeneran en el paso 3 |
| Checkpoint `last.pt` (528.048.707 bytes) | `eval.fetch_checkpoint` lo baja de [Drive](https://drive.google.com/file/d/1j-Y6_RWIFsyos7pAT5eBBhAcPtEjJ95a/view) y verifica el SHA-256 `f693fd7391979856e7059994643e755239276ba5d66a31becf1ebf1e18e93c6b` |
| PNG completos de la evaluación | Se regeneran en el paso 5 |

## Reproducción

Todos los comandos parten de la raíz del repo, salvo que se indique otra cosa. Son
pasos largos: el entrenamiento registrado tardó unas 21 h en CPU y la evaluación genera
72.000 imágenes. Para consultar los resultados no hace falta correr nada.

### 1. Entorno y tests

Con Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r base_code/requirements-eval.txt
OMP_NUM_THREADS=2 python -m pytest base_code/tests -q
```

Resultado esperado: 50 tests pasados y 1 omitido si no hay CUDA. Estas son las versiones
de la evaluación. Las del entrenamiento no quedaron registradas, así que este entorno no
garantiza reproducir el entrenamiento bit a bit.

### 2. Datos

```bash
mkdir -p coco/annotations
curl -L -o coco/train2014.zip http://images.cocodataset.org/zips/train2014.zip
unzip -q coco/train2014.zip -d coco
cd base_code
python -m eval.fetch_coco --root ../coco --count 10000 --seed 0
cd ..
unzip -n coco/annotations_trainval2014.zip -d coco
```

`eval.fetch_coco` baja las anotaciones y las imágenes de val2014 que usa la evaluación.
El último `unzip` asegura que estén también `captions_*` y `person_keypoints_*` de train
y val. No entrenar con train2017: contiene la mayor parte de val2014.

### 3. Features

```bash
python base_code/extract_features.py
python base_code/createCSV.py
python base_code/make_null_embedding.py
```

Leen `coco/` y escriben en `features/train/`. `extract_features.py` tiene fijo
`device = "cpu"`. Según David, los features de la corrida salieron de esta versión del
extractor; no se verificó de forma independiente.

### 4. Entrenamiento

```bash
PYTHONPATH=base_code python train.py --feature-path features/train \
  --max-steps 150000 --global-batch-size 128 --global-seed 0 \
  --lr 1e-4 --cfg-dropout 0.15 --use-amp auto
```

Son los argumentos guardados en el checkpoint. Los resultados quedan en
`results/NNN-DiT-S-4/`. Ver [`README_training.md`](README_training.md).

### 5. Sampling y evaluación

```bash
cd base_code
python -m eval.fetch_checkpoint --output ../checkpoints/david/last.pt
python sample_t2i.py --ckpt ../checkpoints/david/last.pt \
  --prompt "A dog on a beach" --output results/ejemplo \
  --sampler dpm --steps 20 --cfg-scale 4 --seed 0 --batch-size 8 \
  --allow-zero-null --device cpu
```

Los comandos de la evaluación completa (`eval.prepare`, `eval.execute_plan`) y de los
análisis complementarios (`analyze_evaluation.py`) están en
[`base_code/eval/README.md`](base_code/eval/README.md). Los análisis necesitan los PNG
completos, que hay que regenerar antes. `verify_evaluation.py` también los necesita y
depende de las rutas absolutas guardadas en las trazas, así que solo funciona donde esas
rutas existen.

## Problemas conocidos

Se dejaron sin corregir para que el código siga siendo el que produjo esta corrida. El
detalle está en `REPORT_NOTES.md` (§2.1 y §7).

- **∅ en ceros:** `train.py` no le pasa `null_path` al modelo, así que ∅ queda en ceros y
  no es CLIP(""), como pedía la especificación. Para reproducir esta corrida hay que
  mantenerlo; para seguir la especificación hay que corregirlo.
- **`--allow-zero-null`:** `sample_t2i.py` rechaza un ∅ en ceros sin esta opción, y evaluar
  esta corrida la necesita.
- **Reanudar entrenamiento:** `update_ema(ema, model, decay=0)` (`train.py:244`) corre
  después de cargar el checkpoint y pisa el EMA. No afectó a esta corrida, porque no se
  reanudó. Hay que corregirlo antes de usar `--resume`.
- **Comentario incorrecto:** `base_code/dataset_t2i.py:82` dice que los features ya incluyen
  0,18215. La escala se aplica una sola vez, en la línea 84.
