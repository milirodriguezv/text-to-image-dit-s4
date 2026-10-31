# Sampling y evaluación del DiT textual

El pipeline carga pesos EMA del DiT-S/4 y genera imágenes desde captions usando
CLIP ViT-L/14 y el VAE congelado `stabilityai/sd-vae-ft-ema`. Implementa DPM-Solver++
de orden 2 y 20 pasos, DDPM de 250 pasos, FID con clean-fid y CLIP Score L/14+B/32.

## Evaluación entregada

- Checkpoint EMA: 150.000 pasos, 33.003.776 parámetros, batch de entrenamiento 128.
- Seis CFG DPM: 1, 1.5, 2, 3, 4 y 5; 10.000 imágenes por configuración.
- DDPM250: CFG 4, elegido por el menor FID del barrido; 10.000 imágenes.
- FID-2K separado, con CFG 4. No representa una curva entre checkpoints.
- Referencias: COCO val2014, 10.000 IDs fijos, primera caption, seed 0,
  recorte central 256×256. Batch de generación 8. Ejecución final Apple Silicon MPS.

Resultados: [informe](../../results/dit_s4_150k/INFORME.md),
[CSV](../../results/dit_s4_150k/resultados.csv),
[galería](../../results/dit_s4_150k/RESUMEN.html) y
[análisis visual](../../results/dit_s4_150k/ANALISIS_VISUAL.md).
`results/dit_s4_150k/datos/` contiene métricas, trazas comprimidas y manifest congelado;
`figuras/` y `muestras/` contienen gráficas y ejemplos. Los PNG completos y pesos
no se incluyen en Git. Las rutas absolutas de las trazas se conservan como procedencia.

El log de David registra 45.174 imágenes y el extractor selecciona imágenes con
personas, con recorte alrededor de sus cajas. La evaluación no filtra personas y
usa recorte central. Todos los CFG/samplers comparten protocolo, pero sus valores
absolutos también reflejan esa diferencia de datos y recorte. No se han calculado
métricas separadas con/sin personas ni establecido una causa única de los fallos.

## Instalación

Desde la raíz del repositorio, con Python 3.11:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r base_code/requirements-eval.txt
cd base_code
```

Todos los comandos siguientes se ejecutan desde `base_code/`, con el entorno activo.
En una máquina CUDA se necesita una instalación de PyTorch compatible con su GPU.
Los dispositivos explícitos son `cpu`, `mps` o `cuda`; `auto` usa CUDA si está
disponible y CPU en otro caso. CPU y GPU no garantizan igualdad bit a bit.

## Obtener e inspeccionar el checkpoint

```bash
python -m eval.fetch_checkpoint --output ../checkpoints/david/last.pt
python -m eval.inspect_checkpoint --ckpt ../checkpoints/david/last.pt \
  --output results/checkpoint_inspection.json
```

La descarga verifica SHA-256:
`f693fd7391979856e7059994643e755239276ba5d66a31becf1ebf1e18e93c6b`.
El modelo se carga estrictamente desde EMA. El checkpoint guarda `args` y necesita
`torch.load(..., weights_only=False)`; se utiliza el archivo compartido por el equipo.

El checkpoint contiene un buffer nulo **de ceros**, porque el entrenamiento no
cargó `null_path`. Se conserva con `--allow-zero-null`. El `.npy` externo añadido
por David no reemplaza ese buffer ni cambia retroactivamente los pesos.

## Generar ejemplos

```bash
python sample_t2i.py --ckpt ../checkpoints/david/last.pt \
  --prompts-file eval/prompts_david.txt --output results/ejemplos_dpm \
  --sampler dpm --steps 20 --cfg-scale 4 --seed 0 --batch-size 8 \
  --allow-zero-null --device cpu
```

Para DDPM, utiliza `--sampler ddpm --steps 250` y otra carpeta de salida.
Para un texto, sustituye `--prompts-file` por `--prompt "A dog on a beach"`.
Se guarda cada PNG, una vista previa y la correspondencia texto/imagen con metadatos.

## Reproducir el protocolo completo

```bash
python -m eval.fetch_coco --root ../coco --count 10000 --seed 0
python -m eval.prepare \
  --captions ../coco/annotations/captions_val2014.json \
  --images-dir ../coco/val2014 \
  --output results/references_val2014_10k_seed0 \
  --count 10000 --seed 0 --preprocess center-crop

OMP_NUM_THREADS=2 python -u -m eval.execute_plan \
  --ckpt ../checkpoints/david/last.pt \
  --manifest results/references_val2014_10k_seed0/manifest.json \
  --output results/reproduccion_150k \
  --generation-device mps --metric-device mps --clip-device mps \
  --batch-size 8 --metric-batch-size 32 --allow-zero-null
```

Cambia los tres dispositivos a `cuda` o `cpu` según la máquina. Se generan 72.000
imágenes: 2K intermedias, seis configuraciones de 10K y DDPM de 10K. Ejecutar esto
repite el experimento completo; no es necesario para consultar la entrega.
El manifest y checkpoint deben coincidir con los hashes de los resultados entregados.
La descarga no sustituye imágenes ausentes y verifica los IDs frente al split train2014.

Para un protocolo alternativo centrado en personas, `eval.prepare` admite
`--preprocess person-crop --person-annotations /ruta/person_keypoints_val2014.json`.
Eso crea otro conjunto de evaluación y requiere nuevas salidas; no cambia ni
reetiqueta los resultados existentes.

## Evaluación individual y otros checkpoints

```bash
python -m eval.run generate --ckpt ../checkpoints/david/last.pt \
  --manifest results/references_val2014_10k_seed0/manifest.json \
  --output results/otra_evaluacion_2k --count 2000 \
  --sampler dpm --steps 20 --cfg-scale 4 --seed 0 --batch-size 8 \
  --allow-zero-null --device cpu
python -m eval.run metrics --run results/otra_evaluacion_2k/run.json --batch-size 32
```

El paso se lee del checkpoint. El pipeline puede usarse con otros pesos reales,
pero esta entrega solo contiene el de 150K. FID-2K se compara con las mismas 2.000
referencias; no se confunde con FID-10K. `python -m eval.run --help` expone generación,
métricas y barrido CFG; `python -m eval.plot --help` permite gráficas por CFG o paso.

FID usa clean-fid/Inception, modo clean. CLIP usa TorchMetrics sobre cada imagen
con su texto. CLIP Score no es un porcentaje de aciertos y las dos variantes no se
comparan directamente entre sí. No comparar FID entre distintos datasets, recortes,
tamaños de muestra o implementaciones. Una sola semilla no establece significación.

## Integridad y reanudación

`run.json` registra hashes, IDs, captions, semillas, batch, versiones, dispositivo
y huella del código. `metrics.json` queda ligado a esa traza. La huella de los siete
experimentos finales coincide con el código de evaluación incluido; FID-2K conserva
su revisión anterior. No se retocaron los PNG ni se escogieron los ocho ejemplos por calidad.

Para reanudar una generación individual se usa el mismo comando con `--resume`.
Se verifican lotes completos; si cambian código, pesos o configuración hay que usar
otra carpeta. Si solo faltan métricas, ejecuta `eval.run metrics` sin regenerar.
Con los originales completos, desde la raíz del repo:

```bash
.venv/bin/python base_code/verify_evaluation.py \
  --folder /ruta/a/full_evaluation --output /ruta/a/verificacion.json
```

La verificación comprueba integridad y coherencia; no recalcula FID/CLIP.

## Tests

Desde la raíz del proyecto:

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest base_code/tests -q
.venv/bin/python -m pip check
```

Los tests verifican carga, samplers, CFG, semillas, manifest, reanudación, métricas
simuladas e integración. No sustituyen la evaluación real ni descargan modelos.
`eval.build_colab` se conserva como generador de un cuaderno portable y forma parte
de la huella de código del experimento; no se incluyen los diagnósticos ni intentos
parciales de Colab en la entrega de resultados.
