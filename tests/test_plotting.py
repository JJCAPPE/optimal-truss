"""Exercise report exports and verify that animation uses distinct recorded states."""

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest

from optimal_truss.analysis import solve
from optimal_truss.geometry import warren
from optimal_truss.model import BucklingLaw
from optimal_truss.plotting import (
    plot_buckling, plot_comparison, plot_joint, plot_optimization, plot_truss,
    report_figures,
)


def test_complete_report_has_vector_and_raster_exports(tmp_path):
    result = solve(warren(raised_load=True))
    paths = report_figures(result, tmp_path)
    assert len(paths) == 24
    assert {p.suffix for p in paths} == {".png", ".svg", ".pdf"}
    assert all(p.stat().st_size > 1000 for p in paths)
    svg = (tmp_path / "forces.svg").read_text()
    assert "W = 30.00 oz" in svg
    assert "First buckling" in svg
    assert "M11" in svg and "J7" in svg
    assert "Tension (T)" in svg and "Compression (C)" in svg
    with Image.open(tmp_path / "geometry.png") as img:
        assert img.width > 2000 and img.height > 800
    assert not plt.get_fignums()


def test_animation_renders_actual_history_and_comparison(tmp_path):
    base = warren(raised_load=True)
    altered = base.nodes.copy()
    altered[5, 0] += .3
    results = [solve(base), solve(replace(base, nodes=altered))]
    history = [{"generation": i, "nodes": a.truss.nodes.tolist(),
                "failure_load": a.failure_load, "cost": a.cost,
                "objective_value": a.failure_load, "objective": "load",
                "lower_failure_load": a.lower_failure_load}
               for i, a in enumerate(results)]
    paths = plot_optimization(history, base, tmp_path, fps=2, max_frames=2)
    assert len(paths) == 4
    with Image.open(tmp_path / "optimization.gif") as img:
        assert img.n_frames == 2
        img.seek(0)
        first = np.array(img.convert("RGB"))
        img.seek(1)
        last = np.array(img.convert("RGB"))
        assert np.any(first != last)
    assert len(plot_comparison(results, tmp_path)) == 3
    assert not plt.get_fignums()


def test_invalid_plot_requests_are_explicit(tmp_path):
    result = solve(warren(raised_load=True))
    with pytest.raises(ValueError, match="mode"):
        plot_truss(result, tmp_path, mode="not-a-mode")
    with pytest.raises(ValueError, match="joint"):
        plot_joint(result, 99, tmp_path)
    with pytest.raises(ValueError, match="no feasible"):
        plot_optimization([], result.truss, tmp_path)


def test_custom_buckling_law_is_preserved_in_report_figures(tmp_path):
    law = BucklingLaw(coefficient=5000, exponent=2.1, uncertainty_oz=.3)
    result = solve(warren(raised_load=True), buckling=law)
    report_figures(result, tmp_path / "custom")
    fit = (tmp_path / "custom" / "buckling_fit.svg").read_text()
    failure = (tmp_path / "custom" / "forces_at_failure.svg").read_text()
    assert "Fb = 5000" in fit
    assert "±0.3 oz" in fit
    assert f"W = {result.failure_load:.2f} oz" in failure
    assert f"Predicted failure: {result.failure_load:.2f}" in failure
    with pytest.raises(ValueError, match="must match"):
        report_figures(result, tmp_path / "conflicting", BucklingLaw())
    assert not (tmp_path / "conflicting").exists()
    with pytest.raises(ValueError, match="must match"):
        plot_buckling(result, tmp_path / "conflicting", BucklingLaw())


def test_unbounded_capacity_rejected_before_writing_report(tmp_path):
    truss = warren(raised_load=True)
    support_loaded = replace(truss, load_joint=truss.pin)
    result = solve(support_loaded)
    assert np.isinf(result.failure_load)
    destination = tmp_path / "unbounded"
    with pytest.raises(ValueError, match="finite failure load"):
        report_figures(result, destination)
    with pytest.raises(ValueError, match="finite failure load"):
        plot_buckling(result, destination)
    assert not destination.exists()
