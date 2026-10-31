# DiT-S/4 condicionado por texto — MS-COCO

Adaptación de DiT para generar imágenes de 256×256 a partir de captions de COCO.
El modelo utiliza latentes VAE y embeddings CLIP para condicionar la generación
mediante texto. Se conserva la licencia del DiT original en `base_code/LICENSE.txt`.

## Resultados disponibles

Checkpoint EMA de **150.000 pasos**, batch de entrenamiento **128**. El log registra
**45.174 imágenes**, seleccionadas por presencia de personas. La evaluación utiliza
**10.000 referencias de COCO val2014 sin filtrar por personas**, con recorte central.
Esta diferencia de distribución y preprocesado limita la interpretación de las
métricas absolutas y queda documentada en el informe.

| Configuración | FID-10K ↓ | CLIP L/14 ↑ | CLIP B/32 ↑ |
|---|---:|---:|---:|
| DPM-Solver++20, CFG 4 | 71,2168 | 18,2717 | 24,1091 |
| DPM-Solver++20, CFG 5 | 73,2326 | 18,4661 | 24,2080 |
| DDPM250, CFG 4 | 76,6554 | 18,3189 | 24,1666 |

CFG 4 obtiene el menor FID del barrido; CFG 5 obtiene el mayor CLIP. Las imágenes
presentan correspondencia parcial con el texto y defectos de forma/composición.
CLIP Score no es un porcentaje de aciertos. No se han medido intervalos de confianza.

- [Resumen visual y ejemplos](results/dit_s4_150k/RESUMEN.html).
- [Informe completo](results/dit_s4_150k/INFORME.md).
- [Análisis visual](results/dit_s4_150k/ANALISIS_VISUAL.md).
- [Tabla CSV](results/dit_s4_150k/resultados.csv).
- [Uso del pipeline y reproducción](base_code/eval/README.md).
- [Entrenamiento publicado por David](README_training.md).

## Organización

- `base_code/`: arquitectura, datos, sampling, métricas y pruebas.
- `train.py`: entrenamiento textual de David; importa módulos de `base_code/`.
- `log.txt`: log de entrenamiento compartido.
- `results/dit_s4_150k/`: resultados compactos, gráficas, 64 PNG ilustrativos y trazas.
- `REPORT_NOTES.md`: decisiones técnicas del equipo y notas para la memoria.

Los pesos, datasets, cachés, entornos y los PNG completos no se versionan. El código
reutiliza el checkpoint recibido y conserva su condición nula de ceros; no se
sustituye por el embedding externo de texto vacío después de entrenar.

## Instalación y pruebas

Con Python 3.11, desde la raíz del proyecto:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r base_code/requirements-eval.txt
OMP_NUM_THREADS=2 .venv/bin/python -m pytest base_code/tests -q
.venv/bin/python -m pip check
```

Las pruebas validan funcionamiento e integración; las métricas proceden de la
evaluación real. El código original por clases se conserva en `base_code/train.py`,
`sample.py` y `sample_ddp.py`; para el modelo textual se usa `sample_t2i.py`.

Comprobado en esta entrega: **45 tests pasan y uno se omite por requerir CUDA**;
los datos, gráficas y ejemplos compactos coinciden con los originales verificados.
La huella del código de evaluación coincide con los siete experimentos finales.
Se verificaron 72.000 PNG generados y 10.000 referencias en el archivo completo;
esa comprobación no recalcula las métricas.
