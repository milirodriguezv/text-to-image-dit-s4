"""Check the agreed experiment counts and CFG/DDPM selection without generating data."""
from pathlib import Path

from eval.common import read_json, write_json
from sample_t2i import sha256


def test_agreed_plan_runs_2k_six_10k_cfgs_and_ddpm(tmp_path, monkeypatch):
    import eval.execute_plan as plan
    ckpt = tmp_path / "checkpoint.pt"
    manifest = tmp_path / "manifest.json"
    ckpt.write_bytes(b"synthetic test checkpoint")
    manifest.write_text("{}")
    calls = []
    monkeypatch.setattr(plan, "load_manifest", lambda path: ({"preprocess": "center-crop"}, [None] * 10000))

    class Pipeline:
        training = {"step": 150000}

    monkeypatch.setattr(plan.TextToImagePipeline, "from_checkpoint", lambda *a, **k: Pipeline())

    def generate(pipe, source, folder, **settings):
        calls.append(settings)
        Path(folder).mkdir(parents=True, exist_ok=True)
        write_json(Path(folder) / "run.json", settings | {"progress_callback": None})
        settings["progress_callback"](settings["count"], settings["count"])

    def metrics(run, **kwargs):
        config = read_json(run)
        path = Path(run).parent / "metrics.json"
        write_json(path, {"run_sha256": sha256(run), "config": config,
                          "fid": abs(config["cfg_scale"] - 2) + 100,
                          "clip_l14": 20, "clip_b32": 21})
        return path

    monkeypatch.setattr(plan, "generate_run", generate)
    monkeypatch.setattr(plan, "compute_metrics", metrics)
    monkeypatch.setattr(plan, "plot_results", lambda *a, **k: None)
    output = tmp_path / "results"
    plan.execute(ckpt, manifest, output)
    assert [row["count"] for row in calls] == [2000] + [10000] * 7
    assert [row["cfg_scale"] for row in calls[1:7]] == list(plan.CFG_SCALES)
    assert calls[-1]["sampler"] == "ddpm" and calls[-1]["steps"] == 250
    assert calls[-1]["cfg_scale"] == 2
    state = read_json(output / "status.json")
    assert state["status"] == "complete" and len(state["finished_evaluations"]) == 8
