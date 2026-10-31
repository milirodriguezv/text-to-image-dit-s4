"""Build a portable Colab notebook with a code snapshot; no Git push is needed."""
import base64
import hashlib
import io
import json
import textwrap
import zipfile
from pathlib import Path


def build_notebook(destination=None):
    root = Path(__file__).resolve().parents[1]
    destination = Path(destination) if destination else root / "eval" / "colab_evaluation.ipynb"
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if any(part in {"results", "__pycache__", "visuals", "coco", "features", "samples"} for part in relative.parts):
                continue
            if not path.is_file() or path.suffix not in {".py", ".txt", ".md", ".json", ".yml"}:
                continue
            info = zipfile.ZipInfo(str(relative))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    bundle = stream.getvalue()
    encoded = base64.b64encode(bundle).decode()
    digest = hashlib.sha256(bundle).hexdigest()
    cells = []

    def cell(kind, source):
        source = textwrap.dedent(source).strip() + "\n"
        data = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            data.update({"execution_count": None, "outputs": []})
        cells.append(data)

    cell("markdown", """
    # DiT-S/4 — evaluación de Daniel

    Este cuaderno **no entrena**. Evalúa el checkpoint EMA de David a 150.000 pasos.
    Conserva su condición nula de ceros, registrada como desviación del plan inicial.
    Incluye el código de la rama local Daniel: no hace falta publicar esa rama.

    Protocolo: val2014 separado del entrenamiento; FID-10K con clean-fid;
    CLIP Score L/14 y B/32; DPM-Solver++ (20 pasos), seis CFG {1,1.5,2,3,4,5};
    DDPM (250 pasos). Las pruebas pequeñas son diagnósticas, nunca FID-10K.

    1. Selecciona un entorno con GPU en Colab y ejecuta las celdas en orden.
    2. En configuración, indica las rutas reales de COCO y una carpeta nueva de salida.
    3. Revisa las muestras y la estimación de duración antes de iniciar el barrido.
    4. Guarda los resultados en Drive. No cerrar la evaluación alegando solo pruebas pequeñas.

    La GPU y la duración de las sesiones dependen de Colab. No se garantiza acabar
    el barrido en una sesión. Fuente: https://research.google.com/colaboratory/faq.html
    Las celdas de Colab/GPU están preparadas pero no se han ejecutado desde el Mac.
    """)
    cell("code", f'''
    import base64, hashlib, io, json, os, shutil, subprocess, sys, time, zipfile
    from pathlib import Path

    CODE = Path("/content/dit_Daniel/base_code")
    CODE.mkdir(parents=True, exist_ok=True)
    bundle = base64.b64decode({encoded!r})
    assert hashlib.sha256(bundle).hexdigest() == {digest!r}
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        for member in archive.infolist():
            if not (CODE / member.filename).resolve().is_relative_to(CODE.resolve()):
                raise ValueError("Ruta inválida en el paquete de código")
        archive.extractall(CODE)
    print("Código local Daniel cargado; SHA-256 del paquete:", {digest!r})
    ''')
    cell("markdown", """
    ## Dependencias
    Se conservan PyTorch y torchvision del entorno Colab para mantener su CUDA.
    Los demás paquetes siguen requirements-eval.txt. Los comandos de evaluación se
    ejecutan en procesos nuevos, evitando usar módulos antiguos cargados antes de pip.
    """)
    cell("code", '''
    requirements = [line for line in (CODE / "requirements-eval.txt").read_text().splitlines()
                    if line.strip() and not line.startswith("#")
                    and not line.startswith(("torch==", "torchvision=="))]
    subprocess.run([sys.executable, "-m", "pip", "install", *requirements], check=True)
    subprocess.run([sys.executable, "-c", "import torch, torchvision; print('Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'Selecciona un entorno GPU en Colab'"] , check=True)
    ''')
    cell("markdown", """
    ## Drive y configuración
    Autoriza el montaje de tu Drive cuando Colab lo solicite. Modifica las rutas.
    Mantén los 10.000 ejemplos y los seis CFG para la evaluación final acordada.
    El recorte central corresponde al reparto; usa person-crop solo si el equipo ha
    acordado ese cambio y tienes las anotaciones de personas de val2014.
    El checkpoint no permite certificar qué recorte se usó al extraer las features.
    """)
    cell("code", '''
    from google.colab import drive
    drive.mount("/content/drive")

    IMAGES = Path("/content/drive/MyDrive/coco/val2014")  # CAMBIAR si procede
    CAPTIONS = Path("/content/drive/MyDrive/coco/annotations/captions_val2014.json")
    PERSON_ANNOTATIONS = Path("/content/drive/MyDrive/coco/annotations/person_keypoints_val2014.json")
    PREPROCESS = "center-crop"
    EXPORT = Path("/content/drive/MyDrive/dit_eval_Daniel_150k_center_crop")
    WORK = Path("/content/dit_eval_work")
    COUNT = 10000
    CFG_SCALES = [1, 1.5, 2, 3, 4, 5]
    BATCH_SIZE = 8
    METRIC_BATCH_SIZE = 32
    CHECKPOINT = Path("/content/dit_checkpoints/david_last.pt")
    WORK.mkdir(parents=True, exist_ok=True)
    EXPORT.mkdir(parents=True, exist_ok=True)
    MANIFEST = WORK / "references" / "manifest.json"

    def run(*arguments):
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "2"
        subprocess.run([sys.executable, *map(str, arguments)], cwd=CODE, env=env, check=True)

    def restore(folder, name):
        source = EXPORT / f"{name}.zip"
        if not folder.exists() and source.is_file():
            folder.mkdir(parents=True)
            with zipfile.ZipFile(source) as archive:
                for member in archive.infolist():
                    if not (folder / member.filename).resolve().is_relative_to(folder.resolve()):
                        raise ValueError("Ruta inválida en el respaldo")
                archive.extractall(folder)

    def backup(folder, name):
        if folder.is_dir():
            temporary = Path(shutil.make_archive(str(WORK / f"backup_{name}"), "zip", folder))
            shutil.copy2(temporary, EXPORT / f"{name}.zip")
            temporary.unlink()
    ''')
    cell("markdown", """
    ## Checkpoint real y pruebas
    Descarga el archivo público compartido por David y comprueba el hash que se
    verificó localmente. Si Drive cambia el archivo se detiene, sin usar otros pesos.
    El test usa sustitutos para algunos servicios; las celdas posteriores usan los
    modelos preentrenados reales. El reporte de inspección y los tests no son métricas.
    """)
    cell("code", '''
    run("-m", "eval.fetch_checkpoint", "--output", CHECKPOINT)
    run("-m", "eval.inspect_checkpoint", "--ckpt", CHECKPOINT,
        "--output", WORK / "checkpoint_inspection.json")
    shutil.copy2(WORK / "checkpoint_inspection.json", EXPORT / "checkpoint_inspection.json")
    run("-m", "pytest", "tests", "-q")
    ''')
    cell("markdown", """
    ## Ocho muestras y medición de velocidad
    Son los ocho textos fijos del entrenamiento. Guardar estas muestras no sustituye
    la evaluación COCO. El tiempo excluye la carga inicial y no incluye las métricas.
    Si repites esta celda, cambia la carpeta de salida para conservar la medición.
    """)
    cell("code", '''
    SMOKE = WORK / "smoke_dpm20_cfg4"
    run("sample_t2i.py", "--ckpt", CHECKPOINT, "--prompts-file", "eval/prompts_david.txt",
        "--output", SMOKE, "--sampler", "dpm", "--steps", "20", "--cfg-scale", "4",
        "--seed", "0", "--batch-size", BATCH_SIZE, "--device", "cuda", "--allow-zero-null")
    metadata = json.loads((SMOKE / "samples.json").read_text())
    rate = metadata["timing"]["images_per_second"]
    print("Imágenes/s:", rate, "Estimación solo generación DPM (60K):", 60000 / rate / 3600, "horas")
    backup(SMOKE, "smoke_dpm20_cfg4")
    from IPython.display import display, Image
    display(Image(filename=str(SMOKE / "preview.png")))
    ''')
    cell("markdown", """
    ## Referencias COCO val2014
    Si los datos están en un ZIP de Drive, conviene copiarlos y descomprimirlos en
    /content y ajustar IMAGES/CAPTIONS. Evita miles de accesos pequeños a Drive.
    Se prepara una selección reproducible de 10K y se conserva el manifiesto.
    Para repetir una sesión, el ZIP de referencias permite recuperar la selección.
    Fuente sobre almacenamiento: https://research.google.com/colaboratory/faq.html
    """)
    cell("code", '''
    restore(WORK / "references", "references")
    if not MANIFEST.is_file():
        assert IMAGES.is_dir(), f"Falta la carpeta val2014: {IMAGES}"
        assert CAPTIONS.is_file(), f"Faltan las captions: {CAPTIONS}"
        command = ["-m", "eval.prepare", "--captions", CAPTIONS, "--images-dir", IMAGES,
                   "--output", MANIFEST.parent, "--count", COUNT, "--seed", "0",
                   "--preprocess", PREPROCESS]
        if PREPROCESS == "person-crop":
            assert PERSON_ANNOTATIONS.is_file(), "Faltan las anotaciones para person-crop"
            command += ["--person-annotations", PERSON_ANNOTATIONS]
        run(*command)
        backup(MANIFEST.parent, "references")
    manifest_data = json.loads(MANIFEST.read_text())
    assert manifest_data["count"] == COUNT and manifest_data["preprocess"] == PREPROCESS
    print("Manifiesto fijo:", MANIFEST)
    ''')
    cell("markdown", """
    ## FID-2K del checkpoint disponible (150K)
    Esto es la evaluación intermedia, no la final. Los checkpoints de 50K y 100K
    tendrán que entregarlos para evaluar esos pasos; no se pueden reconstruir a
    partir del checkpoint final. Mantener las mismas primeras 2K referencias.
    Se respaldan los resultados al terminar y también si el proceso falla.
    """)
    cell("code", '''
    def evaluate(folder, *, sampler, steps, cfg, count, archive_name):
        restore(folder, archive_name)
        args = ["-m", "eval.run", "generate", "--ckpt", CHECKPOINT,
                "--manifest", MANIFEST, "--output", folder, "--count", count,
                "--sampler", sampler, "--steps", steps, "--cfg-scale", cfg,
                "--seed", "0", "--batch-size", BATCH_SIZE, "--device", "cuda",
                "--allow-zero-null"]
        if folder.exists():
            args += ["--resume"]
        try:
            run(*args)
            run("-m", "eval.run", "metrics", "--run", folder / "run.json",
                "--batch-size", METRIC_BATCH_SIZE, "--device", "cuda")
        finally:
            backup(folder, archive_name)
        return folder / "metrics.json"

    INTERMEDIATE = evaluate(WORK / "step_150000_cfg4_2k", sampler="dpm", steps=20,
                            cfg=4, count=2000, archive_name="step_150000_cfg4_2k")
    print(json.loads(INTERMEDIATE.read_text()))
    ''')
    cell("markdown", """
    ## Evaluación final acordada: FID-10K y ambos CLIP Score para seis CFG
    Se generan **60.000 imágenes**. Cada variante se guarda en un ZIP distinto en
    Drive. No lanzar sin haber revisado la velocidad anterior. Los resultados
    conservan el paso real 150K, batch de entrenamiento 128 y condición nula de ceros.
    Para reanudar, se requieren el mismo código, versiones, GPU y batch size.
    Una interrupción completa del runtime puede perder el trabajo posterior al último
    respaldo; no se garantiza continuidad automática entre GPUs distintas de Colab.
    """)
    cell("code", '''
    FINAL_METRICS = []
    for cfg in CFG_SCALES:
        folder = WORK / f"final_dpm_cfg_{cfg:g}"
        result = evaluate(folder, sampler="dpm", steps=20, cfg=cfg, count=COUNT,
                          archive_name=f"final_dpm_cfg_{cfg:g}")
        FINAL_METRICS.append(result)
    FIGURES = WORK / "figuras_final_dpm"
    run("-m", "eval.plot", "--metrics", *FINAL_METRICS, "--output", FIGURES)
    backup(FIGURES, "figuras_final_dpm")
    display(Image(filename=str(FIGURES / "fid_vs_clip.png")))
    ''')
    cell("markdown", """
    ## Comparación con DDPM (250 pasos)
    Se toma el CFG con menor FID del barrido DPM como criterio explícito de selección,
    se evalúa DDPM con las mismas 10K referencias y se documenta ese criterio.
    Esta celda es más costosa que DPM; no cambiar el número de ejemplos sin etiquetar
    la desviación. Para una prueba corta de funcionamiento usar los ocho prompts,
    pero no presentarla como comparación cuantitativa final.
    """)
    cell("code", '''
    final_results = [json.loads(path.read_text()) for path in FINAL_METRICS]
    best = min(final_results, key=lambda result: result["fid"])
    selected_cfg = best["config"]["cfg_scale"]
    DDPM_METRICS = evaluate(WORK / f"final_ddpm_cfg_{selected_cfg:g}", sampler="ddpm",
                            steps=250, cfg=selected_cfg, count=COUNT,
                            archive_name=f"final_ddpm_cfg_{selected_cfg:g}")
    print("DDPM, CFG elegido por menor FID en DPM:", selected_cfg)
    print(json.loads(DDPM_METRICS.read_text()))
    ''')
    cell("markdown", """
    ## Qué entregar al equipo
    - Manifiesto y protocolo de recorte de las 10K imágenes no usadas para entrenar.
    - FID-2K intermedio de los checkpoints realmente disponibles.
    - Tabla y gráficas FID-10K / CLIP L/14 / CLIP B/32 para los seis CFG.
    - Comparación DDPM y ejemplos visuales.
    - Configuración real: 150K pasos, batch 128, EMA, semillas y versiones.
    - Desviación: el entrenamiento usó ceros como condición nula; no se ha modificado.
    - Limitaciones observadas, sin interpretar la loss de entrenamiento como FID.

    Los ZIP de resultados están en EXPORT. No es necesario subir el código ni los
    pesos a GitHub para ejecutar este cuaderno.
    """)
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}, "colab": {"name": "Evaluación DiT Daniel"},
                             "accelerator": "GPU", "source_bundle_sha256": digest},
                "cells": cells}
    for i, item in enumerate(cells):
        item["id"] = f"daniel-eval-{i:02d}"
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return destination


if __name__ == "__main__":
    print(build_notebook())
