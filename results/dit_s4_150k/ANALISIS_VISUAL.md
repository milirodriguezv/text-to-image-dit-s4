# Revisión visual del checkpoint de 150K

Se muestran los primeros ocho textos del manifest congelado. Esta selección está
fijada por el orden de evaluación; no se han escogido las salidas que parecen mejores.
Los PNG se copian sin retoques. La galería de `RESUMEN.html` permite comparar la
referencia real, DPM20 CFG4 y DDPM250 CFG4, y desplegar los seis CFG de DPM.

Observaciones descriptivas sobre los ejemplos DPM/DDPM de CFG4:

- La descripción de una persona entrando en el mar produce una escena de playa
  reconocible, pero la figura y el objeto que sostiene presentan deformaciones.
- El texto del pájaro entre flores moradas no da un pájaro claramente identificable;
  predominan regiones de color y formas poco definidas en ambos métodos.
- En la descripción de una mujer junto a una jirafa hay formas de personas/caras
  deformadas y el animal no queda claramente resuelto.
- La frase de dos jirafas corriendo entre vegetación no permite identificar las
  dos jirafas ni confirmar la acción; la escena queda dominada por vegetación/formas.
- La descripción de un hombre a caballo observando a otro luchar con una vaca
  produce animales o un rebaño, pero no reproduce de forma clara toda esa relación.

Estos ejemplos muestran correspondencia parcial con categorías o escenarios,
pero fallos de anatomía, número de objetos y acciones complejas. No se presentan
como una medición global ni permiten declarar superioridad visual general de un
sampler. La ligera diferencia de CLIP entre DPM y DDPM no implica que todas las
imágenes de DDPM sean más correctas.

El log de David registra 45.174 imágenes de entrenamiento y el extractor de Linda
selecciona imágenes con personas, con recorte alrededor de sus cajas. La evaluación
no tiene ese filtro: incluye escenas con y sin personas y usa recorte central.
Esa diferencia de distribución y preprocesado puede contribuir a los fallos,
pero estos ocho ejemplos no permiten demostrar su causa ni medir por separado
el rendimiento en escenas con y sin personas.

La referencia COCO es una imagen real asociada al texto. La tarea es generar una
imagen compatible con la descripción, no reconstruir esa fotografía exacta.

## Conclusión que se puede llevar a la memoria

En el checkpoint EMA de 150K evaluado, incrementar CFG de 1 a 4 reduce FID-10K de
98,4370 a 71,2168 y aumenta ambos CLIP Score. Con CFG5, CLIP sigue aumentando,
pero FID sube a 73,2326. Por ello CFG4 es la elección basada en el menor FID dentro
del barrido realizado. DDPM250 con CFG4 obtiene FID 76,6554 frente a 71,2168 de
DPM20, y CLIP ligeramente mayor. Para este experimento, DPM20 CFG4 ofrece el mejor
FID con menos pasos; la calidad visual sigue limitada en las escenas revisadas.

Estas conclusiones se restringen a este checkpoint, este conjunto de referencias,
esta semilla y estas implementaciones. No se ha estimado significación estadística,
una velocidad relativa de 12,5×, una curva de convergencia entre checkpoints ni
una comparación con un baseline externo bajo el mismo protocolo.
