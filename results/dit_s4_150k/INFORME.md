# Evaluación del checkpoint de David

Estado: **completa**. Actualizado: 31/10/2026 11:43:54 (Madrid).

Checkpoint EMA: paso **150,000**. Referencias: COCO val2014, 10.000 imágenes, semilla 0, recorte `center-crop`.

Evaluación terminada: seis CFG de DPM y una comparación DDPM, con 10.000 imágenes por configuración.

## Evaluación intermedia

| Paso | Imágenes | Sampler | CFG | FID-2K ↓ | CLIP L/14 ↑ | CLIP B/32 ↑ |
|---|---|---|---|---|---|---|
| 150000 | 2000 | dpm 20 | 4 | 94.1912 | 18.3290 | 24.1158 |

## Evaluación final: DPM-Solver++ de 20 pasos

| CFG | Imágenes | FID-10K ↓ | CLIP L/14 ↑ | CLIP B/32 ↑ |
|---|---|---|---|---|
| 1 | 10000 | 98.4370 | 15.3453 | 21.0660 |
| 1.5 | 10000 | 87.0201 | 16.2527 | 22.1661 |
| 2 | 10000 | 79.0465 | 16.9160 | 22.9279 |
| 3 | 10000 | 71.9762 | 17.8069 | 23.7768 |
| 4 | 10000 | 71.2168 | 18.2717 | 24.1091 |
| 5 | 10000 | 73.2326 | 18.4661 | 24.2080 |

## Comparación DDPM de 250 pasos

Mismo checkpoint, CFG=4, 10.000 referencias, textos, semilla y batch.

| Sampler | Pasos | FID-10K ↓ | CLIP L/14 ↑ | CLIP B/32 ↑ |
|---|---|---|---|---|
| DPM-Solver++ | 20 | 71.2168 | 18.2717 | 24.1091 |
| DDPM | 250 | 76.6554 | 18.3189 | 24.1666 |

En esta ejecución DPM obtiene menor FID con 20 pasos; DDPM presenta valores de CLIP ligeramente mayores con 250 pasos. No se han realizado repeticiones ni pruebas de significación estadística; los pasos no equivalen a una medición de velocidad.

## Lectura de los resultados

El menor FID-10K del barrido DPM es 71.2168, con CFG=4.
El mayor CLIP Score L/14 es 18.4661, con CFG=5.
Son resultados de este checkpoint, conjunto y semilla; el barrido permite observar el equilibrio entre distribución visual y correspondencia con el texto.

Las gráficas y el CSV del barrido están en `figuras/`. Los ejemplos están en `RESUMEN.html` y `muestras/`.

## Condiciones y limitaciones

- El log de entrenamiento de David registra 45.174 imágenes; el extractor de Linda selecciona imágenes con cajas de personas y realiza un recorte alrededor de ellas. Esta evaluación usa 10.000 imágenes de val2014 sin filtrar por personas y con recorte central. Por tanto, mide el comportamiento sobre un conjunto más amplio y con distinto preprocesado, no exclusivamente sobre el dominio de entrenamiento.
- Todos los CFG y samplers se comparan con los mismos textos y referencias. Sus diferencias son comparables dentro de este protocolo; los valores absolutos también reflejan la diferencia entre entrenamiento y evaluación. No se han calculado métricas separadas con/sin personas ni demostrado que el filtro sea la única causa de los fallos.
- Se conservan los pesos originales y el buffer de ceros que se utilizó como condición vacía durante el entrenamiento. Esto se documenta como desviación del diseño CLIP de texto vacío.
- Se usa el recorte central acordado para las referencias; no se ha alterado el entrenamiento para mejorar la evaluación.
- FID-2K y FID-10K usan tamaños de muestra distintos y no se deben comparar directamente como una misma métrica.
- El alcance de esta entrega es el checkpoint disponible de 150K. No se presenta una curva de evolución entre checkpoints ni resultados de 400K.
- Se evaluó una semilla fija. No se han estimado intervalos de confianza ni comparado contra un baseline externo entrenado y evaluado con el mismo protocolo.
- El checkpoint no contiene los IDs ni el recorte exacto de los datos de entrenamiento. Se verificó que las referencias pertenecen a val2014 y no se solapan con los IDs oficiales de train2014; eso no certifica los archivos concretos utilizados por David.
- `resultados.csv` contiene únicamente evaluaciones completadas; `status.json` conserva el avance y las configuraciones guardan versiones, hashes y dispositivos.

## Revisión visual y texto para la memoria

Véase [ANALISIS_VISUAL.md](ANALISIS_VISUAL.md), con los ocho ejemplos fijos y las conclusiones limitadas al checkpoint evaluado.

## Análisis complementario: referencia real y grupos por personas

Se reutilizan los mismos PNG, captions y checkpoint de DPM-Solver++20, CFG 4.
No se ha entrenado ni generado de nuevo. Se mantienen el recorte central y la
condición nula de ceros del checkpoint.

### CLIP sobre las mismas 10.000 parejas

| Imágenes | CLIP L/14 ↑ | CLIP B/32 ↑ |
|---|---:|---:|
| Reales | 25.5643 | 30.1501 |
| Generadas | 18.2717 | 24.1091 |

### Grupos equilibrados

De los 10.000 IDs, 5.250 tienen al menos una anotación de persona y 4.750 no.
Se toman 4.750 de cada grupo con semilla 0; la tabla global anterior sigue usando
los 10.000 IDs completos. Las fotos y captions conservan su emparejamiento.

| Grupo según anotaciones | N | FID-N ↓ | CLIP L/14 generadas | CLIP L/14 reales | CLIP B/32 generadas | CLIP B/32 reales |
|---|---:|---:|---:|---:|---:|---:|
| Con anotaciones de personas | 4750 | 56.3083 | 19.7638 | 25.5768 | 25.8457 | 30.0862 |
| Sin anotaciones de personas | 4750 | 116.3012 | 16.6117 | 25.5185 | 22.1943 | 30.2021 |

En esta selección, el grupo con anotaciones obtiene menor FID y mayor CLIP generado
en ambas variantes. Los CLIP de las fotos reales son similares entre grupos.
Esto es compatible con un rendimiento distinto por tipo de escena, pero no prueba
que el filtro del entrenamiento sea su causa: los grupos también difieren en
contenido, captions y composición.

La etiqueta procede de la foto original y no garantiza que la persona sea visible
en el recorte central o que se mencione en la caption. FID-4750 no se compara
directamente con FID-10K; CLIP real es una referencia, no un techo ni un porcentaje
de calidad. Una sola selección y semilla no establecen significación estadística.

Los controles reprodujeron las métricas globales previas: diferencias absolutas
menores de 0,000004 en CLIP y 0,000013 en FID. Los PNG se verificaron antes y después.

[Protocolo, controles y tabla completa](analisis_complementario/INFORME.md),
[CSV de grupos](analisis_complementario/metricas_grupos.csv) e
[IDs seleccionados](analisis_complementario/ids_grupos.json).
