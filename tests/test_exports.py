import json

import numpy as np
import pytest
from scipy.io import loadmat

from optimal_truss.analysis import solve
from optimal_truss.geometry import warren
from optimal_truss.io import export_analysis, read_truss
from optimal_truss.model import BucklingLaw


def test_matlab_export_roundtrip_and_independent_force_assembly(tmp_path):
    truss = warren(raised_load=True)
    a = solve(truss)
    summary = export_analysis(a, tmp_path)
    data = loadmat(tmp_path / "design.mat")
    n, m = len(truss.nodes), len(truss.members)
    assert data["C"].shape == (n, m)
    assert data["X"].shape == data["Y"].shape == (1, n)
    assert data["Sx"].shape == data["Sy"].shape == (n, 3)
    assert data["L"].shape == (2*n, 1)
    np.testing.assert_array_equal(data["C"].sum(axis=0), 2)
    # Rebuild from saved connectivity and coordinates, as an external grader would.
    matrix = np.zeros((2*n, m+3))
    xy = np.column_stack((data["X"].ravel(), data["Y"].ravel()))
    for member in range(m):
        i, j = np.flatnonzero(data["C"][:, member])
        unit = (xy[j] - xy[i]) / np.linalg.norm(xy[j] - xy[i])
        matrix[[i, i+n], member] = unit
        matrix[[j, j+n], member] = -unit
    matrix[:n, m:] = data["Sx"]
    matrix[n:, m:] = data["Sy"]
    forces = np.linalg.solve(matrix, data["L"]).ravel()
    np.testing.assert_allclose(forces, np.r_[a.forces, a.reactions], atol=1e-10)
    assert summary["passes_all_checks"]
    assert read_truss(tmp_path / "geometry.json").to_dict() == truss.to_dict()
    assert (tmp_path / "cut_plan.csv").exists()


def test_noncompliant_design_is_not_reported_as_passing(tmp_path):
    export_analysis(solve(warren(raised_load=True)), tmp_path)
    assert (tmp_path / "cut_plan.csv").exists()
    summary = export_analysis(solve(warren()), tmp_path)
    assert not summary["passes_all_checks"]
    assert "Constraint status: FAIL" in (tmp_path / "analysis.md").read_text()
    exported = json.loads((tmp_path / "summary.json").read_text())
    assert any(c["name"] == "load_height" and not c["passed"] for c in exported["checks"])
    assert not (tmp_path / "cut_plan.csv").exists()


def test_export_retains_custom_fit_and_rejects_mixed_provenance(tmp_path):
    law = BucklingLaw(coefficient=3000, exponent=2, uncertainty_oz=2)
    a = solve(warren(raised_load=True), buckling=law)
    summary = export_analysis(a, tmp_path)
    assert summary["buckling_law"]["coefficient"] == 3000
    assert summary["buckling_law"]["uncertainty_oz"] == 2
    with pytest.raises(ValueError, match="must match"):
        export_analysis(a, tmp_path, buckling=BucklingLaw())


def test_no_compression_does_not_create_a_misleading_report(tmp_path):
    truss = warren(raised_load=True)
    truss.load_joint = truss.pin
    with pytest.raises(ValueError, match="No finite buckling"):
        export_analysis(solve(truss), tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
