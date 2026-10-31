import json

import pytest

from report_evaluation import export_report


def test_report_keeps_unfinished_cfgs_pending(tmp_path):
    row = {"name": "intermediate", "count": 2000, "sampler": "dpm", "steps": 20,
           "cfg_scale": 4, "fid": 99., "clip_l14": 20., "clip_b32": 21.}
    state = {"status": "running", "training_step": 150000, "preprocess": "center-crop",
             "finished_evaluations": [row]}
    (tmp_path / "status.json").write_text(json.dumps(state))
    assert export_report(tmp_path) == "running"
    assert (tmp_path / "report.md").read_text().count("| Pendiente | Pendiente | Pendiente |") == 6
    assert len((tmp_path / "results.csv").read_text().splitlines()) == 2
    state["status"] = "complete"
    (tmp_path / "status.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="six"):
        export_report(tmp_path)
