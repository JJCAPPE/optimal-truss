"""Optimization checks use feasible geometry and recomputed physical results."""

import numpy as np
import pytest

from optimal_truss.analysis import solve
from optimal_truss.constraints import validate
from optimal_truss.geometry import warren
from optimal_truss.model import Constraints, Truss
from optimal_truss.optimize import optimize_geometry


def test_search_keeps_feasible_non_decreasing_incumbents():
    baseline = warren(raised_load=True, load_height=1.25)
    result = optimize_geometry(baseline, generations=3, population=2, seed=4)
    assert result.analysis.failure_load >= solve(baseline).failure_load
    values = [item["objective_value"] for item in result.history]
    assert np.all(np.diff(values) >= 0)
    assert len(result.history) == result.metadata["generations_completed"] + 1
    assert result.metadata["evaluations"] == result.metadata["solver_nfev"] + 1
    for item in result.history:
        truss = Truss(
            name="saved generation", nodes=np.asarray(item["nodes"]),
            members=baseline.members, pin=baseline.pin, roller=baseline.roller,
            load_joint=baseline.load_joint,
        )
        analysis = solve(truss)
        assert all(check.passed for check in validate(truss, analysis))
        assert analysis.failure_load == pytest.approx(item["failure_load"])


def test_fixed_seed_reproduces_geometry_and_trace():
    options = dict(generations=2, population=1, seed=23, objective="load-cost")
    first = optimize_geometry(warren(raised_load=True, load_height=1.25), **options)
    second = optimize_geometry(warren(raised_load=True, load_height=1.25), **options)
    np.testing.assert_array_equal(first.truss.nodes, second.truss.nodes)
    assert first.history == second.history
    assert first.metadata["best_objective_value"] == pytest.approx(first.analysis.load_cost)


def test_noncompliant_comparison_truss_cannot_seed_search():
    with pytest.raises(ValueError, match="feasible starting geometry"):
        optimize_geometry(warren(), generations=1, population=1)


def test_callback_and_lower_envelope_objective():
    records = []
    result = optimize_geometry(
        warren(raised_load=True, load_height=1.25), generations=2, population=1,
        objective="lower-load", callback=records.append,
    )
    assert records == result.history
    assert records[-1]["objective_value"] == pytest.approx(result.analysis.lower_failure_load)
    assert "not a proven global optimum" in result.metadata["scope"]


def test_reversed_supports_and_fixed_span_bounds():
    baseline = warren(raised_load=True, load_height=1.25)
    reversed_supports = Truss(
        name="Right-hand pin", nodes=baseline.nodes, members=baseline.members,
        pin=baseline.roller, roller=baseline.pin, load_joint=baseline.load_joint,
    )
    limits = Constraints(span_min=30, span_max=30)
    result = optimize_geometry(
        reversed_supports, generations=2, population=1, seed=5, limits=limits
    )
    assert all(check.passed for check in validate(result.truss, result.analysis, limits))
    assert result.analysis.failure_load >= solve(reversed_supports).failure_load
    assert abs(result.truss.nodes[result.truss.roller, 0]
               - result.truss.nodes[result.truss.pin, 0]) == pytest.approx(30)
    assert result.metadata["buckling_law"]["coefficient"] == 3391.349


@pytest.mark.parametrize("options", [
    {"objective": "magic"}, {"generations": 0}, {"population": 0},
    {"seed": -1}, {"seed": True},
])
def test_invalid_search_options(options):
    with pytest.raises(ValueError):
        optimize_geometry(warren(raised_load=True), **options)
