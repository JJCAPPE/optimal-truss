"""Independent equilibrium checks, failure limits, geometry, and stock packing."""

import numpy as np
import pytest
from scipy.io import loadmat, savemat

from optimal_truss.analysis import equilibrium_matrix, matlab_arrays, solve, solve_equilibrium
from optimal_truss.constraints import stock_cut_plan, validate
from optimal_truss.geometry import warren
from optimal_truss.model import BucklingLaw, Constraints, Truss


def triangle():
    return Truss("Asymmetric analytical example", [[0, 0], [4, 0], [1, 3]],
                 [[0, 1], [0, 2], [1, 2]], 0, 1, 2)


def failed_checks(truss, analysis=None, limits=Constraints()):
    return {check.name for check in validate(truss, analysis, limits) if not check.passed}


def test_asymmetric_triangle_matches_hand_equilibrium():
    # Moments about the pin: Ry_right*4 = 30*1. Each support joint then
    # determines its diagonal force; pin-x equilibrium gives bottom tension.
    result = solve(triangle(), 30)
    np.testing.assert_allclose(result.reactions, [0, 22.5, 7.5], atol=1e-12)
    np.testing.assert_allclose(result.forces, [7.5, -7.5 * np.sqrt(10), -7.5 * np.sqrt(2)])
    assert result.residual < 1e-12


def test_symmetric_triangle_independent_solution():
    truss = Truss("Equilateral analytical example", [[0, 0], [10, 0], [5, 5 * np.sqrt(3)]],
                  [[0, 1], [0, 2], [1, 2]], 0, 1, 2)
    result = solve(truss, 30)
    np.testing.assert_allclose(result.forces, [5 * np.sqrt(3), -10 * np.sqrt(3), -10 * np.sqrt(3)])
    np.testing.assert_allclose(result.reactions, [0, 15, 15], atol=1e-12)
    assert result.critical_members == (1, 2)


def test_arbitrary_horizontal_and_multiple_nodal_loads():
    # At apex: +6 horizontal and -30 vertical. An extra -5 load at the pin
    # changes only its vertical reaction. Moment balance gives right Ry=12.
    forces, reactions = solve_equilibrium(triangle(), [[0, -5], [0, 0], [6, -30]])
    np.testing.assert_allclose(reactions, [-6, 23, 12])
    np.testing.assert_allclose(forces, [12, -6 * np.sqrt(10), -12 * np.sqrt(2)])


def test_arbitrary_load_array_must_match_joint_count():
    with pytest.raises(ValueError, match="finite"):
        solve_equilibrium(triangle(), [[0, -30]])


def test_load_scaling_and_unit_force_invariance():
    one, doubled = solve(triangle(), 12), solve(triangle(), 24)
    np.testing.assert_allclose(doubled.forces, 2 * one.forces)
    np.testing.assert_allclose(doubled.reactions, 2 * one.reactions)
    np.testing.assert_allclose(doubled.unit_forces, one.unit_forces)
    assert doubled.failure_load == one.failure_load


def test_failure_and_uncertainty_are_derived_from_compression_only():
    law = BucklingLaw(coefficient=1000, exponent=2, uncertainty_oz=2)
    result = solve(triangle(), buckling=law)
    # Left diagonal length sqrt(10), compression/total load sqrt(10)/4.
    # Right diagonal length sqrt(18), compression/total load sqrt(2)/4.
    assert result.failure_load == pytest.approx(400 / np.sqrt(10))
    assert result.critical_members == (1,)
    assert result.uncertainty == pytest.approx(8 / np.sqrt(10))
    assert result.lower_failure_load == pytest.approx(392 / np.sqrt(10))
    assert result.upper_failure_load == pytest.approx(408 / np.sqrt(10))
    at_failure = solve(triangle(), result.failure_load, law)
    assert abs(at_failure.forces[1]) == pytest.approx(result.capacities[1])
    assert abs(at_failure.forces[2]) < result.capacities[2]


def test_fit_band_envelope_can_change_critical_member():
    result = solve(triangle(), buckling=BucklingLaw(1000, 2, 50))
    assert result.critical_members == (1,)
    # The right diagonal becomes limiting at the lower strength band.
    assert result.lower_failure_load == pytest.approx((1000 / 18 - 50) / (np.sqrt(2) / 4))
    assert result.lower_failure_load < result.failure_load - result.uncertainty


def test_analysis_retains_custom_buckling_fit():
    law = BucklingLaw(coefficient=4200, exponent=1.9, uncertainty_oz=2.7)
    result = solve(triangle(), buckling=law)
    assert result.buckling is law
    np.testing.assert_allclose(result.capacities, law.capacity(result.lengths))


def test_load_at_support_has_no_finite_buckling_prediction():
    truss = triangle()
    truss.load_joint = truss.pin
    result = solve(truss)
    np.testing.assert_allclose(result.forces, 0)
    assert np.isinf(result.failure_load)
    assert result.critical_members == ()
    assert result.uncertainty == 0
    assert {"load_off_supports", "minimum_load"} <= failed_checks(truss, result)


def test_matrix_equilibrium_and_matlab_roundtrip(tmp_path):
    truss = triangle()
    data = matlab_arrays(truss, 30)
    assert data["C"].shape == (3, 3)
    np.testing.assert_array_equal(data["C"].sum(axis=0), [2, 2, 2])
    assert data["Sx"].shape == data["Sy"].shape == (3, 3)
    assert data["X"].shape == data["Y"].shape == (1, 3)
    assert data["L"].shape == (6, 1)
    assert data["L"][5, 0] == 30
    path = tmp_path / "design.mat"
    savemat(path, data)
    saved = loadmat(path)
    a = equilibrium_matrix(truss)
    solution = np.linalg.solve(a, saved["L"]).ravel()
    result = solve(truss)
    np.testing.assert_allclose(solution, np.r_[result.forces, result.reactions])
    assert result.residual == np.max(np.abs(a @ np.r_[result.forces, result.reactions] - data["L"].ravel()))


def test_modified_warren_default_is_fully_compliant():
    truss = warren(raised_load=True)
    assert truss.nodes[truss.load_joint, 1] == 1.25
    result = solve(truss)
    assert not failed_checks(truss, result)
    assert result.failure_load > 30


def test_exact_warren_remains_available_but_flagged():
    truss = warren()
    result = solve(truss)
    np.testing.assert_allclose(result.lengths, 10)
    assert {"load_height", "stock_cutting"} <= failed_checks(truss, result)


def test_total_material_is_not_enough_to_prove_cutting_feasibility():
    assert 11 * 10 < 120
    assert stock_cut_plan([10] * 11) is None
    result = solve(warren(raised_load=True))
    plan = stock_cut_plan(result.lengths)
    assert sorted(i for bar in plan for i in bar) == list(range(11))
    for bar, capacity in zip(plan, (48, 48, 24)):
        assert sum(result.lengths[bar]) <= capacity + 1e-7


def test_kerf_can_make_a_cut_plan_infeasible():
    assert stock_cut_plan([12, 12], [24]) == [[0, 1]]
    assert stock_cut_plan([12, 12], [24], kerf_in=0.01) is None


def test_count_alone_does_not_prove_stability():
    truss = Truss("Collinear", [[0, 0], [1, 0], [2, 0]], [[0, 1], [1, 2], [0, 2]], 0, 2, 1)
    assert "member_count" not in failed_checks(truss)
    assert "stable" in failed_checks(truss)
    with pytest.raises(ValueError, match="singular|unstable"):
        solve(truss)


def test_crossing_members_are_rejected():
    truss = Truss("Crossed", [[0, 0], [10, 0], [10, 10], [0, 10]],
                  [[0, 1], [1, 2], [2, 3], [0, 2], [1, 3]], 0, 1, 2)
    assert "planar_members" in failed_checks(truss)


def test_member_passing_through_another_joint_is_rejected():
    truss = Truss("Through joint", [[0, 0], [5, 0], [10, 0], [5, 10]],
                  [[0, 2], [0, 3], [2, 3], [1, 3], [0, 1]], 0, 2, 1)
    assert "planar_members" in failed_checks(truss)


def test_joint_directly_below_load_is_rejected():
    truss = warren(raised_load=True)
    truss.nodes[4, 0] = truss.nodes[1, 0]
    truss.load_joint = 4
    assert "load_clearance" in failed_checks(truss)


@pytest.mark.parametrize("change, message", [
    ({"members": [[0, 1], [1, 0], [0, 2]]}, "duplicated"),
    ({"members": [[0, 1], [1, 7], [0, 2]]}, "does not exist"),
    ({"members": [[0, 1], [1.5, 2], [0, 2]]}, "integer"),
    ({"pin": 1, "roller": 1}, "different"),
    ({"nodes": [[0, 0], [float("nan"), 0], [1, 2]]}, "finite"),
])
def test_malformed_input_errors(change, message):
    data = triangle().to_dict()
    data.update(change)
    with pytest.raises(ValueError, match=message):
        Truss.from_dict(data)


def test_geometry_json_roundtrip_preserves_precision():
    original = warren(raised_load=True)
    restored = Truss.from_dict(original.to_dict())
    assert restored.to_dict() == original.to_dict()


@pytest.mark.parametrize("load", [0, -1, float("nan")])
def test_invalid_applied_load(load):
    with pytest.raises(ValueError, match="positive"):
        solve(triangle(), load)


def test_class_fit_uses_supplied_units_and_precision():
    law = BucklingLaw()
    assert law.capacity([10])[0] == pytest.approx(3391.349 * 10 ** -2.012)
    assert law.uncertainty_oz == 1.52


@pytest.mark.parametrize("limits", [
    {"kerf_in": -0.1}, {"kerf_in": float("nan")},
    {"member_min": 15, "member_max": 14}, {"stock_lengths": [0, 48]},
])
def test_malformed_constraint_limits(limits):
    with pytest.raises(ValueError):
        Constraints(**limits)
