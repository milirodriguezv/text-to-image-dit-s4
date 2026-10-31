# Análisis complementario — DiT-S/4 EMA 150K

Se reutilizan las 10.000 imágenes reales y las 10.000 generadas con DPM-Solver++20, CFG 4. No se ha entrenado ni generado de nuevo.

## CLIP de las imágenes reales y generadas (10.000 parejas)

| Imágenes | CLIP L/14 ↑ | CLIP B/32 ↑ |
|---|---:|---:|
| Reales | 25.5643 | 30.1501 |
| Generadas | 18.2717 | 24.1091 |

CLIP real es una referencia del mismo conjunto y captions; no es un techo teórico ni permite convertir el resultado generado en un porcentaje de calidad.

## Grupos equilibrados: 4,750 imágenes por grupo

| Grupo por anotaciones | N | FID-N ↓ | CLIP L/14 generadas | CLIP L/14 reales | CLIP B/32 generadas | CLIP B/32 reales |
|---|---:|---:|---:|---:|---:|---:|
| Con anotaciones de personas | 4750 | 56.3083 | 19.7638 | 25.5768 | 25.8457 | 30.0862 |
| Sin anotaciones de personas | 4750 | 116.3012 | 16.6117 | 25.5185 | 22.1943 | 30.2021 |

## Protocolo y alcance

- Dentro de las 10K referencias hay 5250 IDs con anotaciones y 4750 sin ellas. La selección se realiza sobre IDs ordenados, con `random.Random(0)`, y se guarda en `ids_grupos.json`.
- La etiqueta se obtiene de al menos una anotación en `person_keypoints_val2014.json`, sin exigir keypoints visibles. Describe la foto original: no garantiza una persona visible tras el recorte central ni que la caption hable de personas. Sin anotaciones no es una inspección visual de ausencia de personas.
- Se mantienen los PNG originales y el recorte central de la evaluación general. No se reproduce aquí el recorte de personas del entrenamiento.
- FID-N usa solo las N reales y las N generadas de los mismos IDs de cada grupo, con clean-fid/clean/Inception-v3. No debe compararse directamente con FID-10K; igualar N no elimina la incertidumbre ni las diferencias de composición entre grupos.
- Los resultados son descriptivos para una selección y una semilla; no prueban una causa de los fallos ni significación estadística.
- CLIP usa los mismos modelos y procesamiento que la evaluación original. Se guardan scores por imagen antes del clamp; cada media se calcula como `max(mean(100*cos), 0)`, igual que TorchMetrics 1.9. Los valores de L/14 y B/32 se interpretan por separado.
- Se comprobó la reproducción de las métricas globales originales antes de aceptar este análisis. Las cachés de features no se versionan.

## Archivos

`resultados.json` conserva métricas, controles y procedencia; `metricas_grupos.csv` permite incorporar la tabla al informe; `scores_por_imagen.csv` permite auditar las medias; `ids_grupos.json` fija las selecciones.
