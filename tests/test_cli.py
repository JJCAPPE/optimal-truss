import json

from optimal_truss.cli import main
from optimal_truss.geometry import warren


def test_generate_analyze_and_noncompliant_exit_status(tmp_path):
    path = tmp_path / "design.json"
    assert main(["generate", "--out", str(path)]) == 0
    assert main(["analyze", str(path), "--out", str(tmp_path / "results"), "--no-figures"]) == 0
    summary = json.loads((tmp_path / "results/summary.json").read_text())
    assert summary["passes_all_checks"]
    assert main(["generate", "--kind", "equilateral", "--out", str(path)]) == 0
    assert main(["analyze", str(path), "--out", str(tmp_path / "reference"), "--no-figures"]) == 2


def test_demo_records_real_run_and_reproducible_geometry(tmp_path):
    assert main(["demo", "--out", str(tmp_path), "--generations", "2", "--population", "2",
                 "--seeds", "42", "--no-figures"]) == 0
    history = json.loads((tmp_path / "optimization/history.json").read_text())
    summary = json.loads((tmp_path / "optimization/best/summary.json").read_text())
    assert len(history) == 3
    assert history[-1]["failure_load"] >= history[0]["failure_load"]
    assert summary["passes_all_checks"]
    assert summary["nominal_failure_load_oz"] == history[-1]["failure_load"]


def test_invalid_search_load_does_not_start_a_run(tmp_path):
    assert main(["demo", "--out", str(tmp_path), "--load", "nan", "--no-figures"]) == 1
    assert not (tmp_path / "baseline").exists()


def test_no_improvement_does_not_rename_the_input(tmp_path):
    from argparse import Namespace
    from optimal_truss.cli import _optimize

    baseline = warren(raised_load=True)
    original = baseline.to_dict()
    args = Namespace(seeds=[4], generations=1, population=1, objective="load",
                     kerf=0, load=30, team="Test", section="A2", no_figures=True)
    _optimize(baseline, tmp_path, args)
    assert baseline.to_dict() == original
