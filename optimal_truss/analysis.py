"""Joint-equilibrium solver and first-member buckling prediction.

The model neglects self-weight, bending, tensile failure, joint failure, and
out-of-plane instability. A static capacity is not evidence of a 60-second test.
"""

from dataclasses import dataclass

import numpy as np

from .model import BucklingLaw, Truss


@dataclass
class Analysis:
    truss: Truss
    load_oz: float
    lengths: np.ndarray
    forces: np.ndarray
    reactions: np.ndarray
    unit_forces: np.ndarray
    capacities: np.ndarray
    failure_load: float
    uncertainty: float
    critical_members: tuple
    lower_failure_load: float
    upper_failure_load: float
    cost: float
    load_cost: float
    residual: float
    condition: float
    buckling: BucklingLaw = BucklingLaw()


def equilibrium_matrix(truss: Truss):
    """Return A for A @ [member forces, pin-x, pin-y, roller-y] = -external.

    Rows are all joint x equations followed by all joint y equations. Positive
    member force means tension: the force points from that joint toward its
    member's other endpoint. Assembly is O(M); dense solution is O(J**3).
    """
    n, m = len(truss.nodes), len(truss.members)
    a = np.zeros((2 * n, m + 3))
    vectors = truss.nodes[truss.members[:, 1]] - truss.nodes[truss.members[:, 0]]
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError("A member has zero length")
    directions = vectors / lengths[:, None]
    for k, (i, j) in enumerate(truss.members):
        a[i, k], a[j, k] = directions[k, 0], -directions[k, 0]
        a[n + i, k], a[n + j, k] = directions[k, 1], -directions[k, 1]
    a[truss.pin, m] = 1
    a[n + truss.pin, m + 1] = 1
    a[n + truss.roller, m + 2] = 1
    return a


def matlab_arrays(truss: Truss, load_oz=30.0):
    """Return the course's C, Sx, Sy, X, Y, L arrays for scipy.io.savemat.

    L is the right-hand side of equilibrium, so a downward applied force has
    a POSITIVE entry. Sx/Sy columns are pin-x, pin-y, roller-y. X/Y are rows.
    """
    n, m = len(truss.nodes), len(truss.members)
    c, sx, sy = np.zeros((n, m)), np.zeros((n, 3)), np.zeros((n, 3))
    for k, (i, j) in enumerate(truss.members):
        c[[i, j], k] = 1
    sx[truss.pin, 0] = 1
    sy[truss.pin, 1], sy[truss.roller, 2] = 1, 1
    load = np.zeros((2 * n, 1))
    load[n + truss.load_joint, 0] = load_oz
    return {"C": c, "Sx": sx, "Sy": sy,
            "X": truss.nodes[:, 0][None, :], "Y": truss.nodes[:, 1][None, :], "L": load}


def _solve_equilibrium(truss, external_loads):
    external = np.asarray(external_loads, dtype=float)
    if external.shape != truss.nodes.shape or not np.isfinite(external).all():
        raise ValueError("External loads must be a finite [J, 2] array of [Fx, Fy]")
    a = equilibrium_matrix(truss)
    if a.shape[0] != a.shape[1]:
        raise ValueError("A determinate truss requires M = 2J - 3")
    condition = float(np.linalg.cond(a))
    if not np.isfinite(condition) or condition > 1e12:
        raise ValueError("Equilibrium matrix is singular or numerically unstable")
    rhs = -np.r_[external[:, 0], external[:, 1]]
    try:
        forces = np.linalg.solve(a, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Truss is unstable: equilibrium has no unique solution") from exc
    return forces, condition, a, rhs


def solve_equilibrium(truss: Truss, external_loads):
    """Return (member forces, reactions) for arbitrary signed nodal loads.

    external_loads is a J-by-2 array in oz-force: positive Fx is right and
    positive Fy is up. Multiple loaded joints and horizontal loads are allowed.
    Reaction order is pin-x, pin-y, roller-y; positive member force is tension.
    This function solves forces only, so it is also suitable for independently
    checking a practice problem with a different loading arrangement.
    """
    forces, _, _, _ = _solve_equilibrium(truss, external_loads)
    m = len(truss.members)
    return forces[:m], forces[m:]


def solve(truss: Truss, load_oz=30.0, buckling=BucklingLaw()):
    """Solve a downward load and predict first buckling using unit-load scaling.

    failure_load = min(Fb / abs(unit_force)) over compressed members. Uncertainty
    propagates the fit band through the nominal critical member; for ties the
    largest tied-member band is reported. The lower/upper limits separately
    minimize across ALL compressed members, allowing the critical member to
    switch. These are deterministic fit-band envelopes, not a probabilistic
    95% confidence interval for the whole truss.
    """
    if not np.isfinite(load_oz) or load_oz <= 0:
        raise ValueError("Applied load must be finite and positive")
    external = np.zeros_like(truss.nodes)
    external[truss.load_joint, 1] = -1.0
    unit, condition, a, rhs = _solve_equilibrium(truss, external)
    m = len(truss.members)
    lengths = np.linalg.norm(truss.nodes[truss.members[:, 1]] - truss.nodes[truss.members[:, 0]], axis=1)
    capacities = buckling.capacity(lengths)
    compression = np.flatnonzero(unit[:m] < -1e-10)
    if len(compression):
        ratios = -unit[compression]
        limits = capacities[compression] / ratios
        failure = float(np.min(limits))
        tied = np.isclose(limits, failure, rtol=1e-6, atol=1e-8)
        critical = tuple(int(i) for i in compression[tied])
        uncertainty = float(np.max(buckling.uncertainty_oz / ratios[tied]))
        lower = float(np.min(np.maximum(capacities[compression] - buckling.uncertainty_oz, 0) / ratios))
        upper = float(np.min((capacities[compression] + buckling.uncertainty_oz) / ratios))
    else:
        failure, lower, upper, uncertainty, critical = np.inf, np.inf, np.inf, 0.0, ()
    cost = float(np.sum(lengths) + 10 * len(truss.nodes))
    forces = unit * load_oz
    return Analysis(truss, float(load_oz), lengths, forces[:m], forces[m:], unit[:m], capacities,
                    failure, uncertainty, critical, lower, upper, cost, failure / cost,
                    float(np.max(np.abs(a @ forces - rhs * load_oz))), condition, buckling)
