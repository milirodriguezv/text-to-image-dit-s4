# Plan de trabajo — Persona 2: Arquitectura DiT + Text Conditioning + CFG

> Responsable: Persona 2 (rama `mili`) · Entregable: `base_code/models.py` (+ tests)
> Fecha del plan: 06/10 · Fin de Fase 2 del proyecto: 15/10

Este documento explica **qué** vamos a hacer, **en qué orden** y **por qué**, para que cualquier
integrante del grupo pueda entender cómo funciona el condicionamiento por texto del modelo y qué
espera recibir/entregar cada parte.

---

## Índice

0. [Resumen en 1 minuto (para contarle al grupo)](#0-resumen-en-1-minuto-para-contarle-al-grupo)
1. [Conceptos necesarios](#1-conceptos-necesarios)
2. [Dónde encaja mi parte en el proyecto](#2-dónde-encaja-mi-parte-en-el-proyecto)
3. [Plan secuencial](#3-plan-secuencial)
   - [Parte A — Preparación del entorno](#parte-a--preparación-del-entorno)
   - [Parte B — Contrato de interfaz (stub) y comunicación al grupo](#parte-b--contrato-de-interfaz-stub-y-comunicación-al-grupo)
   - [Parte C — `TextEmbedder`](#parte-c--textembedder)
   - [Parte D — Adaptar `DiT.__init__` e `initialize_weights`](#parte-d--adaptar-dit__init__-e-initialize_weights)
   - [Parte E — `forward` y `forward_with_cfg`](#parte-e--forward-y-forward_with_cfg)
   - [Parte F — Tests](#parte-f--tests)
   - [Parte G — Documentación (`REPORT_NOTES.md`)](#parte-g--documentación-report_notesmd)
   - [Parte H — Entrega, merge e integración](#parte-h--entrega-merge-e-integración)
   - [Parte I — Durante el entrenamiento y la evaluación](#parte-i--durante-el-entrenamiento-y-la-evaluación)
4. [Calendario propuesto](#4-calendario-propuesto)
5. [Dependencias y riesgos](#5-dependencias-y-riesgos)
6. [Checklist final](#6-checklist-final)

---

## 0. Resumen en 1 minuto (para contarle al grupo)

- El DiT original genera imágenes de **una clase de ImageNet** (un número del 0 al 999). Ese número
  se convierte en un vector con una tabla (`LabelEmbedder`) y ese vector "dirige" a la red.
- Nosotros queremos generar imágenes **a partir de una frase**. La frase la pasa CLIP (congelado) a un
  vector de 768 números, `e_pooled`.
- Mi trabajo es **sustituir la tabla de clases por una pequeña red (MLP) que proyecta el vector de
  CLIP (768) al tamaño del modelo (384)**. El resto del DiT queda igual.
- Además, para poder usar **Classifier-Free Guidance (CFG)** al generar, el modelo tiene que aprender
  también a generar *sin* texto. Por eso, durante el entrenamiento, el 15 % de las veces cambiamos la
  frase por el vector de la frase vacía, `∅ = CLIP("")`.
- Al generar, el modelo se ejecuta dos veces (con texto y con ∅) y combinamos ambas predicciones para
  que la imagen se ajuste más al texto.
- El cambio total es pequeño (~60–80 líneas en `models.py`), pero está **en el camino crítico**: sin
  él no se puede entrenar.

---

## 1. Conceptos necesarios

### 1.1 Qué predice el modelo

El DiT es un *denoiser*: recibe un latente con ruido `z_t` (4×32×32), el instante `t` (0–999) y la
condición `y`, y devuelve **8 canales**:

| Canales | Nombre | Para qué sirve |
|---|---|---|
| 0–3 | `ε_θ` | Predicción del ruido que hay que quitar. Se entrena con MSE |
| 4–7 | `Σ_θ` | Varianza de cada paso inverso (`learn_sigma=True`). Se entrena con el término `vb` de la loss |

Por eso la salida es `(B, 8, 32, 32)` aunque la entrada sea `(B, 4, 32, 32)`. `training_losses`
(en `diffusion/`, que no se toca) **exige** esos 8 canales.

### 1.2 Cómo entra la condición en la red: adaLN-Zero

```
 t ──► TimestepEmbedder ──► t_emb (384) ─┐
                                          ├─(+)──► c (384)
 e_pooled (768) ──► TextEmbedder ──► y_emb (384) ─┘
                                                  │
                    en cada uno de los 12 bloques ▼
          c ──► SiLU ──► Linear ──► (β1, γ1, α1, β2, γ2, α2)   (6 vectores de 384)

          x = x + α1 · Attn( LN(x)·(1+γ1) + β1 )
          x = x + α2 · MLP ( LN(x)·(1+γ2) + β2 )
```

- `c` **no se mira con atención**: modula (escala y desplaza) los canales de **todos** los 64 tokens
  por igual. Es un condicionamiento *global* (Variante A).
- **"Zero"**: la última `Linear` que produce α, β, γ se inicializa a **cero**. Entonces, al empezar,
  α = 0 y cada bloque es la identidad → el entrenamiento es estable desde el paso 0. La capa final
  también se inicializa a cero, así que **al inicio la salida del modelo es exactamente 0**. Esto es
  algo que *no* debemos romper y que vamos a testear.

### 1.3 Por qué una MLP y no una tabla

- Una clase es un número discreto → basta una tabla de búsqueda (`nn.Embedding`).
- Un texto es un vector continuo de 768 dimensiones → necesitamos una función aprendible que lo lleve a
  384. Usamos `Linear(768→384) → SiLU → Linear(384→384)`, la misma forma que el `TimestepEmbedder`, así
  que el texto y el tiempo se tratan de forma simétrica (igual que SD3 con su vector CLIP *pooled*).
- Coste: 443,136 parámetros (~1.3 % del modelo).

### 1.4 El embedding nulo `∅` y el text dropout

- En el DiT original, la "no-clase" es una fila extra **aprendida** de la tabla (la clase 1000).
- Con texto existe un "vacío" natural: la frase vacía `""`. Usamos `∅ = CLIP("")`, **congelado**
  (`register_buffer`, no es un parámetro, el optimizador no lo toca).
- Durante el entrenamiento, con probabilidad 15 % sustituimos `e_pooled` por `∅`. Así el mismo modelo
  aprende a la vez:
  - `ε_θ(z_t, t, texto)` → predicción **condicional**
  - `ε_θ(z_t, t, ∅)` → predicción **incondicional**
- El dropout solo se aplica en `model.train()`. En `model.eval()` (EMA, sampling) nunca se tira el texto
  salvo que se fuerce con `force_drop_ids`.

### 1.5 Classifier-Free Guidance (CFG)

Al generar, combinamos ambas predicciones con una escala `s`:

```
ε = ε_uncond + s · (ε_cond − ε_uncond)
```

- `s = 1` → solo condicional (sin guía).
- `s > 1` → "empuja" la imagen en la dirección del texto: más fidelidad al prompt (CLIP Score sube),
  menos diversidad (FID suele empeorar a partir de cierto punto). Por eso hacemos el barrido
  `s ∈ {1, 1.5, 2, 3, 4, 5}`.
- **Eficiencia:** las dos predicciones se calculan en **un solo forward** con el batch duplicado:
  `[z, z]` con `y = [e_prompt, ∅]`.
- **Detalle del repo original:** aplica la guía solo a 3 de los 4 canales de `ε` (por
  reproducibilidad con sus resultados). Nosotros usamos los **4 canales** (fórmula estándar). Los
  canales `Σ_θ` no se guían.

---

## 2. Dónde encaja mi parte en el proyecto

```
Persona 1 (datos)            Persona 2 (YO, modelo)        Persona 4 (sampling/eval)
 e_pooled.npy ─────────┐      models.py ──────────┐          sample_t2i.py, eval/
 null_empty_string.npy ┼────► TextEmbedder (∅)    │               ▲
 latents.npy           │                          ▼               │
                       └────────────────────► Persona 3 ──► checkpoint (con ∅ dentro)
                                              train.py
```

### Contrato de interfaz (lo que los demás esperan de `models.py`)

| Quién | Cómo usa el modelo |
|---|---|
| Persona 3 (`train.py`) | `model = DiT_models["DiT-S/4"](input_size=32, text_dim=768, class_dropout_prob=0.15, null_path=...)` |
| Persona 3 (`training_losses`) | `model(x_t, t, y=e_pooled)` — el kwarg **tiene que** llamarse `y` |
| Persona 3 (EMA) | `deepcopy(model)` copia también el buffer `∅` (correcto: es constante) |
| Persona 4 (DPM-Solver++) | `model(cat([z, z]), t, y=cat([e_prompt, e_null]))[:, :4]` y aplica CFG fuera |
| Persona 4 (DDPM 250) | `model.forward_with_cfg(x, t, y, cfg_scale)` y se queda con `samples.chunk(2)[0]` |
| Persona 4 (∅) | `e_null = model.y_embedder.null_embedding` (sale del checkpoint, no necesita el `.npy`) |

| Tensor | Forma | dtype |
|---|---|---|
| `x` | `(B, 4, 32, 32)` | float32 (o bf16 dentro de autocast) |
| `t` | `(B,)` | long, valores 0–999 |
| `y` | `(B, 768)` | float32 |
| salida | `(B, 8, 32, 32)` | float32 |

---

## 3. Plan secuencial

### Parte A — Preparación del entorno

**Objetivo:** poder importar y ejecutar `models.py` en local (CPU/MPS; no hace falta GPU).

1. Rama de trabajo: `mili` (ya existe). Mantener los cambios visibles como diff respecto a `main`.
2. Instalar dependencias mínimas:
   ```bash
   pip install timm numpy pytest
   pip install transformers   # solo para generar ∅ provisional (Parte C.4)
   ```
3. Comprobar que el `models.py` original importa y construye `DiT-S/4`:
   ```python
   from models import DiT_models
   m = DiT_models["DiT-S/4"](input_size=32)
   print(sum(p.numel() for p in m.parameters()))   # 32,945,024 (original)
   ```

**Hecho cuando:** el modelo original se instancia sin errores.

---

### Parte B — Contrato de interfaz (stub) y comunicación al grupo

**Objetivo:** que las Personas 3 y 4 puedan programar contra mi API **antes** de que yo termine.

1. Acordar con el grupo (mensaje corto + enlace a la sección 2 de este documento):
   - Nombre de los argumentos del constructor: `text_dim=768`, `class_dropout_prob=0.15`,
     `null_path=None`.
   - `null_path` es **opcional**: el buffer `∅` se guarda dentro del `state_dict`, así que al cargar un
     checkpoint ya viene incluido. Solo hace falta pasarlo al **crear el modelo para entrenar**.
   - Cómo obtener ∅ en sampling: `model.y_embedder.null_embedding`.
2. Subir un primer commit con la **firma final** (aunque la lógica esté a medias) para que nadie tenga
   que adivinar.

> **No especificado en `CLAUDE.md`:** el nombre `null_path` ni si es obligatorio. Es una decisión
> nuestra; dejarla registrada en `REPORT_NOTES.md`.

**Hecho cuando:** Personas 3 y 4 confirman la firma.

---

### Parte C — `TextEmbedder`

**Objetivo:** reemplazar `LabelEmbedder` (`models.py:67-94`) por un embedder de texto con la misma
estructura (mismos nombres de métodos y misma lógica de dropout).

Esquema previsto:

```python
class TextEmbedder(nn.Module):
    """
    Projects CLIP pooled text embeddings into vector representations. Also handles text dropout
    (replacement by the frozen null embedding) for classifier-free guidance.
    """
    def __init__(self, text_dim, hidden_size, dropout_prob, null_path=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.dropout_prob = dropout_prob
        null = torch.zeros(1, text_dim)
        if null_path is not None:
            null = torch.from_numpy(np.load(null_path)).float().reshape(1, text_dim)
        self.register_buffer("null_embedding", null)   # frozen ∅ = CLIP(""), saved in state_dict

    def token_drop(self, e, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(e.shape[0], device=e.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(drop_ids[:, None], self.null_embedding.to(e.dtype), e)

    def forward(self, e, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            e = self.token_drop(e, force_drop_ids)
        return self.proj(e)
```

Puntos clave:

1. **Mismo criterio de dropout que el original:** solo si `(train and p > 0)` o si se fuerza.
2. **El dropout se hace sobre el vector de 768 (antes de la MLP)**, sustituyendo por ∅. Así la MLP ve
   ∅ como una entrada más y aprende a proyectarlo.
3. **`∅` es buffer, no parámetro:** no se entrena, no cuenta en el número de parámetros, viaja en el
   checkpoint y la EMA lo copia vía `deepcopy`.
4. **∅ provisional:** mientras la Persona 1 no entregue `null_empty_string.npy`, lo genero yo con
   `CLIPTokenizer` + `CLIPTextModel` (`openai/clip-vit-large-patch14`, `padding="max_length"`,
   `max_length=77`, `pooler_output`). Es exactamente lo que hará `extract_features.py`, así que el
   valor coincidirá (hasta precisión fp16).
5. **Protección contra ∅ = 0 por error:** si `null_path=None`, el buffer queda a ceros. Eso está bien al
   cargar un checkpoint (se sobrescribe), pero sería un bug silencioso al **entrenar**. Propuesta:
   pedir a la Persona 3 un `assert model.y_embedder.null_embedding.abs().sum() > 0` en `train.py`.

---

### Parte D — Adaptar `DiT.__init__` e `initialize_weights`

1. `DiT.__init__` (`models.py:148-178`):
   - Sustituir `num_classes=1000` por `text_dim=768` y añadir `null_path=None`.
   - `self.y_embedder = TextEmbedder(text_dim, hidden_size, class_dropout_prob, null_path)`.
   - **No tocar** `learn_sigma=True`, `PatchEmbed`, `pos_embed`, bloques ni `FinalLayer`.
   - Mantener el nombre `class_dropout_prob` (así lo usan `train.py` y `CLAUDE.md`), aunque ahora sea
     dropout de texto.
2. `initialize_weights` (`models.py:180-213`):
   - **Borrar** `nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)`.
   - **Añadir** `normal_(std=0.02)` en `y_embedder.proj[0].weight` y `y_embedder.proj[2].weight`
     (igual que el `t_embedder`). Los bias quedan a 0 por `_basic_init`.
   - **Mantener** los zero-init de adaLN y de la capa final (adaLN-Zero).
3. Actualizar el docstring de `forward`: `y: (N, 768) tensor of CLIP pooled text embeddings`.
4. Las configuraciones `DiT_XL_2` … `DiT_S_8` no cambian: reenvían `**kwargs`.

**Nota sobre el número de parámetros:** el reparto y `CLAUDE.md` dicen "≈33.4M", pero el cálculo
capa por capa da:

| Modelo | Parámetros |
|---|---|
| DiT-S/4 original (tabla 1001×384) | 32,945,024 (≈32.9M, coincide con el paper) |
| DiT-S/4 con `TextEmbedder` | **33,003,776 (≈33.0M)** |

(Incluye `pos_embed`, 24,576, que es `nn.Parameter` congelado. `∅` no cuenta porque es buffer.)
**El objetivo correcto del test es ≈33.0M**; hay que corregirlo en `CLAUDE.md`, en el reparto y en
`REPORT_NOTES.md`.

---

### Parte E — `forward` y `forward_with_cfg`

1. **`forward(x, t, y)`**: no cambia. Sigue siendo `c = t_embedder(t) + y_embedder(y, self.training)`.
2. **`forward_with_cfg(x, t, y, cfg_scale)`** (`models.py:250-263`):
   - Cambiar `eps, rest = model_out[:, :3], model_out[:, 3:]` por
     `eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]` (4 canales).
   - Actualizar el comentario del repo explicando el cambio.
   - El resto se mantiene: usa solo la primera mitad de `x` duplicada, devuelve la `ε` guiada en ambas
     mitades y `Σ_θ` sin guiar. Quien llama se queda con `samples.chunk(2)[0]`.
   - **Convención de `y`:** `y = cat([e_prompt, e_null.expand(n, -1)])`, primero la mitad condicional.

Cómo funciona `forward_with_cfg` paso a paso:

```
entrada:  x = [z, z]          y = [e_prompt, ∅]           (2n muestras)
          half = x[:n]  → combined = [half, half]          (garantiza el mismo z en ambas mitades)
          model_out = forward(combined, t, y)              (2n, 8, 32, 32)
          eps = model_out[:, :4]     rest = model_out[:, 4:]
          cond_eps, uncond_eps = eps[:n], eps[n:]
          guided = uncond + s·(cond − uncond)
salida:   cat([ [guided, guided], rest ], dim=1)           (2n, 8, 32, 32)
```

#### Explicación intuitiva de la Parte E (para el grupo)

**1. Qué predice el modelo en cada paso.** Para generar una imagen partimos de ruido puro y, en cada
paso, el modelo responde a "¿qué parte de esto es ruido, para poder quitarlo?". Esa respuesta es `ε`.

**2. CFG = pedirle dos opiniones y exagerar la diferencia.** En cada paso le pedimos dos respuestas
sobre la misma imagen:
- **Con texto:** "quitá el ruido pensando en *un perro en la playa*" → `ε_cond`
- **Sin texto (∅):** "quitá el ruido pensando en *una imagen cualquiera*" → `ε_uncond`

La diferencia `ε_cond − ε_uncond` es "lo que aporta el texto". CFG la **amplifica** con la escala `s`:

```
ε = ε_uncond + s · (ε_cond − ε_uncond)
     imagen        cuánto empujamos   lo que el texto
     genérica      hacia el texto     cambia
```

- `s = 1` → respuesta normal con texto, sin exagerar.
- `s = 4` → "hacelo 4 veces más parecido a lo que pide el texto".

Es como subir el volumen de "lo que dice el texto" por encima del ruido de fondo.

**3. El truco del batch doble.** Para no ejecutar el modelo dos veces, metemos las dos preguntas en un
solo lote: imágenes `[z, z]`, textos `["perro", ∅]`. Una sola pasada devuelve ambas respuestas;
`forward_with_cfg` separa las mitades, aplica la fórmula y devuelve el resultado.

**4. El bug que corregimos.** El latente tiene **4 canales** (4 "capas" abstractas que juntas forman la
imagen, parecido a RGB pero con 4). El ruido `ε` también tiene 4 canales. El código original aplica el
"volumen extra" del texto **solo a 3 de los 4**:

```
canal 0  → guiado con s ✓
canal 1  → guiado con s ✓
canal 2  → guiado con s ✓
canal 3  → SIN guía (siempre s = 1) ✗
```

Es como subir el volumen de 3 de 4 parlantes. Los autores de DiT lo dejaron así a propósito para
reproducir exactamente los números de su paper (lo avisan en un comentario). Nosotros usamos la CFG
estándar: si no, en el barrido de `s` (1, 1.5, 2, 3, 4, 5) un cuarto de la imagen no se estaría
guiando. **La corrección es cambiar el `3` por `4`** (`self.in_channels`).

**5. Lo que no se toca.** Los otros 4 canales de salida, `Σ`, dicen **cuánta aleatoriedad agregar** en
cada paso. No se amplifican con CFG; se usan tal cual (igual que en DiT y GLIDE).

> **En una frase:** la Parte E hace que la guía por texto (CFG) se aplique a **los 4 canales** del ruido
> y no solo a 3.

**Resultado (06/10):** implementado en el commit `a304609`. Con pesos aleatorios, la fórmula coincide
con el cálculo manual para `s ∈ {0, 1, 4}`, el canal 3 ahora está guiado (con el código viejo
coincidía con el condicional sin guiar), `Σ` sale sin guiar y el muestreador DDPM del repo corre sin NaN.

---

### Parte F — Tests

Archivo propuesto: `base_code/tests/test_models.py` (pytest, CPU, entradas sintéticas
`torch.randn(B, 768)`, sin datos reales).

| # | Test | Cómo | Esperado |
|---|---|---|---|
| 1 | Nº de parámetros | `sum(p.numel() for p in model.parameters())` | 33,003,776 |
| 2 | Forma de salida | `model(randn(2,4,32,32), randint(0,1000,(2,)), randn(2,768))` | `(2, 8, 32, 32)` |
| 3 | Zero-output al inicio | misma llamada | todos los valores == 0 |
| 4 | ∅ es buffer congelado | `"y_embedder.null_embedding"` en `state_dict()` y **no** en `named_parameters()` | sí / no |
| 5 | ∅ se carga del `.npy` | crear un `.npy` temporal y comparar | idéntico |
| 6 | Dropout ≈ 15 % | `model.train()`, llamar `token_drop` sobre N=100,000 vectores y contar filas == ∅ | 0.15 ± 0.01 |
| 7 | Sin dropout en eval | `model.eval()`, `y_embedder(e, train=False)` == `proj(e)` | idénticos |
| 8 | `force_drop_ids` | `force_drop_ids=[1,0]` → fila 0 == `proj(∅)`, fila 1 == `proj(e)` | sí |
| 9 | El texto cambia la salida | **re-inicializar aleatoriamente** `final_layer` y adaLN (al inicio la salida es 0 para cualquier `y`), y comparar salidas con dos `y` distintos | distintas |
| 10 | CFG con 4 canales | con pesos aleatorios: `forward_with_cfg` vs cálculo manual `uncond + s·(cond−uncond)` sobre `[:, :4]` | iguales; `[:, 4:]` sin guiar |
| 11 | CFG con s = 1 | `forward_with_cfg(..., 1.0)[:n, :4]` == salida condicional | iguales |
| 12 | Compatibilidad con `training_losses` | `create_diffusion("").training_losses(model, z0, t, dict(y=e))` | devuelve `loss`, `mse`, `vb` de forma `[B]`, sin error |
| 13 | EMA copia ∅ | `deepcopy(model).y_embedder.null_embedding` == original | sí |
| 14 | bf16 (opcional, si hay GPU) | forward bajo `torch.autocast(dtype=bfloat16)` | sin error, salida finita |

> El test 9 es importante: como la salida inicial es 0, **un test ingenuo de "condicional ≠
> incondicional" pasaría por casualidad o fallaría siempre**. Hay que romper el zero-init *solo en
> el test*.

---

### Parte G — Documentación (`REPORT_NOTES.md`)

Según las reglas del proyecto, cualquier cambio de comportamiento o desviación se registra:

1. **§5 Sanity checks:** rellenar "Param count" (33,003,776), "Output at init" y "Measured text
   dropout", con fecha.
2. **§4 Correcciones:** añadir la fila "≈33.4M → ≈33.0M" con la cuenta.
3. **§7 Observaciones:** decisiones de interfaz (`null_path` opcional, ∅ dentro del checkpoint, nombre
   `class_dropout_prob` mantenido) y la recomendación del assert de ∅ ≠ 0.
4. Corregir "≈33.4M" en `CLAUDE.md` y en `reparto_trabajo_DiT_detallado.md` (avisar al grupo).

---

### Parte H — Entrega, merge e integración

1. Commit(s) en `mili` con mensajes claros:
   - `models: replace LabelEmbedder with TextEmbedder (adaLN-Zero-Text)`
   - `models: apply CFG on all 4 eps channels`
   - `tests: model shape, params, zero-init, dropout, CFG`
2. Abrir PR hacia `main` **coordinado con la Persona 3**: al mergear mi `models.py`, el `train.py`,
   `sample.py` y `sample_ddp.py` originales dejan de funcionar (usan `num_classes`). Lo ideal es que
   mi PR y el de `train.py` se mergeen juntos o el mío justo antes del suyo.
3. Apoyar a la Persona 3 en el **overfit test** (64 imágenes, 1 caption, sin flip, dropout 0,
   ~2K pasos). Cómo interpretar los fallos que me afectan:
   - Misma imagen para todas las captions → el condicionamiento no está conectado (mi parte).
   - Imagen equivocada para una caption → filas desalineadas (Persona 1).
   - Ruido de color → problema de escalado/normalización (Persona 1 / 4).
4. Comprobar con la Persona 3 el dropout medido en el entrenamiento real (≈15 %).

---

### Parte I — Durante el entrenamiento y la evaluación

Mientras corren los 400K pasos (~35 h estimadas), puedo:

- Apoyar a la Persona 4 con `forward_with_cfg` y el uso de ∅ en DPM-Solver++ / DDPM.
- Redactar la sección de **método** del informe (adaLN-Zero-Text, TextEmbedder, ∅, CFG): gran parte
  está ya en `REPORT_NOTES.md` §3.4–3.6 y en la sección 1 de este documento.
- Preparar (si sobra tiempo) la ablación opcional **MLP vs Linear** en la proyección de texto.

---

## 4. Calendario propuesto

| Día | Partes | Resultado |
|---|---|---|
| Mar 06/10 | A, B | Entorno listo; firma acordada y publicada en `mili` |
| Mar 06/10 – Mié 07/10 | C, D, E | `models.py` implementado |
| Mié 07/10 | F, G | Tests en verde; `REPORT_NOTES.md` actualizado; PR abierto |
| Jue 08/10 | H | Merge coordinado + overfit test con Persona 3 |
| Vie 09/10 → ~Dom 11/10 | I | Entrenamiento completo (Persona 3) |
| Lun 12/10 → Jue 15/10 | I | Evaluación (Persona 4), redacción del informe |

**Estado al 06/10:** Partes A–G completadas en `mili` (commits `76118fb` stub de interfaz,
`f65bf53` lógica de `TextEmbedder`, `869ec46` init de la proyección + dropout 0.15, `a304609` CFG en
4 canales, `457e9fe` tests). Pendiente: confirmación de la firma por el grupo, PR/merge (Parte H) y
overfit test con la Persona 3.

> El plan original de `CLAUDE.md` preveía empezar el entrenamiento completo hacia el 03/10; ya vamos
> con retraso. Como mi parte bloquea el entrenamiento, la prioridad es cerrar A–F cuanto antes.

---

## 5. Dependencias y riesgos

| Dependencia / riesgo | Tipo | Mitigación |
|---|---|---|
| `null_empty_string.npy` (Persona 1) | Blanda | Generar ∅ provisional con CLIP; el definitivo será el mismo vector |
| Firma del constructor (Personas 3 y 4) | **Estricta** (contrato) | Parte B: publicarla el primer día |
| Merge rompe `train.py`/`sample.py` originales | Coordinación | Merge conjunto con la Persona 3 |
| ∅ queda a ceros si se olvida `null_path` al entrenar | Bug silencioso | Assert en `train.py` |
| Objetivo de parámetros mal especificado (33.4M) | Documentación | Corregir a ≈33.0M |
| Test de condicionamiento con pesos zero-init | Falso resultado | Re-inicializar pesos solo en el test |
| `torch.load` en PyTorch ≥ 2.6 (`weights_only=True`) | Afecta a 3 y 4 | Avisar: usar `weights_only=False` o guardar `vars(args)` |
| No modificar `diffusion/` | Regla | Solo se toca `models.py` y se añaden tests |

---

## 6. Checklist final

- [x] `timm` instalado; el `models.py` original importa
- [x] Firma publicada en `mili` (pendiente confirmación del grupo) (`text_dim`, `class_dropout_prob`, `null_path`)
- [x] `TextEmbedder` implementado (MLP + ∅ buffer + dropout + `force_drop_ids`)
- [x] `DiT.__init__` y `initialize_weights` adaptados; zero-init intacto
- [x] `forward_with_cfg` con 4 canales
- [x] Tests 1–13 en verde (14 si hay GPU) — 19 passed + 1 skipped (bf16 CUDA, sin GPU local)
- [x] `REPORT_NOTES.md` actualizado (§4, §5, §7); corregido "33.4M" en `CLAUDE.md` y en el reparto
- [ ] PR abierto y merge coordinado con la Persona 3
- [ ] Overfit test superado (con la Persona 3)
