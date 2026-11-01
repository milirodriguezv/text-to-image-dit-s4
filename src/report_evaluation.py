"""Export completed experiment results to a readable report and CSV while a run proceeds."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

SCALES = (1, 1.5, 2, 3, 4, 5)


def export_report(folder):
    folder = Path(folder)
    state = json.loads((folder / "status.json").read_text())
    rows = state.get("finished_evaluations", [])
    final = {row["cfg_scale"]: row for row in rows if row["sampler"] == "dpm" and row["count"] == 10000}
    ddpm = [row for row in rows if row["sampler"] == "ddpm" and row["count"] == 10000]
    if state["status"] == "complete" and (set(final) != set(SCALES) or not ddpm):
        raise ValueError("Cannot report completion without six FID-10K CFGs and DDPM")
    temporary = folder / "results.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["checkpoint_step", "evaluation", "count", "sampler", "steps", "cfg_scale", "fid", "clip_l14", "clip_b32"])
        for row in rows:
            writer.writerow([state.get("training_step"), row["name"], row["count"], row["sampler"], row["steps"],
                             row["cfg_scale"], row["fid"], row["clip_l14"], row["clip_b32"]])
    temporary.replace(folder / "results.csv")
    now = datetime.now(ZoneInfo("Europe/Madrid")).strftime("%d/%m/%Y %H:%M:%S")
    lines = ["# Evaluación del checkpoint de David", "",
             f"Estado: **{'completa' if state['status'] == 'complete' else 'en curso' if state['status'] == 'running' else state['status']}**. Actualizado: {now} (Madrid).", "",
             f"Checkpoint EMA: paso **{state.get('training_step', 150000):,}**. Referencias: COCO val2014, 10.000 imágenes, semilla 0, recorte `{state['preprocess']}`.", "",
             ("Evaluación terminada: seis CFG de DPM y una comparación DDPM, con 10.000 imágenes por configuración."
              if state["status"] == "complete" else
              f"En ejecución: `{state.get('evaluation', '')}`; fase `{state.get('phase', '')}`; {state.get('images_done', 0):,}/{state.get('images_target', 0):,} imágenes."), "",
             "## Evaluación intermedia", "", "| Paso | Imágenes | Sampler | CFG | FID-2K ↓ | CLIP L/14 ↑ | CLIP B/32 ↑ |",
             "|---|---|---|---|---|---|---|"]
    for row in rows:
        if row["count"] == 2000:
            lines.append(f"| {state.get('training_step')} | 2000 | {row['sampler']} {row['steps']} | {row['cfg_scale']:g} | {row['fid']:.4f} | {row['clip_l14']:.4f} | {row['clip_b32']:.4f} |")
    lines += ["", "## Evaluación final: DPM-Solver++ de 20 pasos", "",
              "| CFG | Imágenes | FID-10K ↓ | CLIP L/14 ↑ | CLIP B/32 ↑ |", "|---|---|---|---|---|"]
    for cfg in SCALES:
        row = final.get(cfg)
        lines.append(f"| {cfg:g} | 10000 | {row['fid']:.4f} | {row['clip_l14']:.4f} | {row['clip_b32']:.4f} |" if row
                     else f"| {cfg:g} | 10000 | Pendiente | Pendiente | Pendiente |")
    lines += ["", "## Comparación DDPM de 250 pasos", ""]
    if ddpm:
        row = ddpm[0]
        counterpart = final.get(row["cfg_scale"])
        lines += [f"Mismo checkpoint, CFG={row['cfg_scale']:g}, 10.000 referencias, textos, semilla y batch.", "",
                  "| Sampler | Pasos | FID-10K ↓ | CLIP L/14 ↑ | CLIP B/32 ↑ |",
                  "|---|---|---|---|---|"]
        if counterpart:
            lines.append(f"| DPM-Solver++ | 20 | {counterpart['fid']:.4f} | {counterpart['clip_l14']:.4f} | {counterpart['clip_b32']:.4f} |")
        lines.append(f"| DDPM | 250 | {row['fid']:.4f} | {row['clip_l14']:.4f} | {row['clip_b32']:.4f} |")
        if counterpart:
            lines += ["", "En esta ejecución DPM obtiene menor FID con 20 pasos; DDPM presenta valores de CLIP ligeramente mayores con 250 pasos. No se han realizado repeticiones ni pruebas de significación estadística; los pasos no equivalen a una medición de velocidad."]
    else:
        lines.append("Pendiente. Se usará el CFG con menor FID-10K del barrido DPM, con las mismas 10.000 referencias.")
    if len(final) == len(SCALES):
        best_fid = min(final.values(), key=lambda row: row["fid"])
        best_clip = max(final.values(), key=lambda row: row["clip_l14"])
        lines += ["", "## Lectura de los resultados", "",
                  f"El menor FID-10K del barrido DPM es {best_fid['fid']:.4f}, con CFG={best_fid['cfg_scale']:g}.",
                  f"El mayor CLIP Score L/14 es {best_clip['clip_l14']:.4f}, con CFG={best_clip['cfg_scale']:g}.",
                  "Son resultados de este checkpoint, conjunto y semilla; el barrido permite observar el equilibrio entre distribución visual y correspondencia con el texto.", "",
                  "Las gráficas y el CSV del barrido están en `figuras/`. Las vistas previas de cada configuración están en su carpeta."]
    lines += ["", "## Condiciones y limitaciones", "",
              "- El log de entrenamiento de David registra 45.174 imágenes; el extractor de Linda selecciona imágenes con cajas de personas y realiza un recorte alrededor de ellas. Esta evaluación usa 10.000 imágenes de val2014 sin filtrar por personas y con recorte central. Por tanto, mide el comportamiento sobre un conjunto más amplio y con distinto preprocesado, no exclusivamente sobre el dominio de entrenamiento.",
              "- Todos los CFG y samplers se comparan con los mismos textos y referencias. Sus diferencias son comparables dentro de este protocolo; los valores absolutos también reflejan la diferencia entre entrenamiento y evaluación. No se han calculado métricas separadas con/sin personas ni demostrado que el filtro sea la única causa de los fallos.",
              "- Se conservan los pesos originales y el buffer de ceros que se utilizó como condición vacía durante el entrenamiento. Esto se documenta como desviación del diseño CLIP de texto vacío.",
              "- Se usa el recorte central acordado para las referencias; no se ha alterado el entrenamiento para mejorar la evaluación.",
              "- FID-2K y FID-10K usan tamaños de muestra distintos y no se deben comparar directamente como una misma métrica.",
              "- El alcance de esta entrega es el checkpoint disponible de 150K. No se presenta una curva de evolución entre checkpoints ni resultados de 400K.",
              "- Se evaluó una semilla fija. No se han estimado intervalos de confianza ni comparado contra un baseline externo entrenado y evaluado con el mismo protocolo.",
              "- El checkpoint no contiene los IDs ni el recorte exacto de los datos de entrenamiento. Se verificó que las referencias pertenecen a val2014 y no se solapan con los IDs oficiales de train2014; eso no certifica los archivos concretos utilizados por David.",
              "- `results.csv` contiene únicamente evaluaciones completadas; `status.json` conserva el avance y las configuraciones guardan versiones, hashes y dispositivos.", ""]
    temporary = folder / "report.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(folder / "report.md")
    return state["status"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--follow", action="store_true", help="Keep exports updated until this single execution finishes")
    args = parser.parse_args()
    while True:
        status = export_report(args.folder)
        if not args.follow or status != "running":
            break
        time.sleep(20)


if __name__ == "__main__":
    main()
