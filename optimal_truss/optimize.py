"""Reproducible geometry search for one specified, simple-truss topology.

Differential evolution proposes joint coordinates; the statics solver evaluates
them. There is no learned surrogate or training data. Every saved incumbent
passes the same independent checks used for the report. Connectivity, the load
joint, and the support joints stay fixed, so this is not a topology search or a
certificate of a global optimum.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution

from .analysis import Analysis, solve
from .constraints import validate
from .model import BucklingLaw, Constraints, Truss


@dataclass
class OptimizationResult:
    """Best feasible geometry, its analysis, and the complete generation trace."""

    truss: Truss
    analysis: Analysis
    history: list[dict]
    metadata: dict


def _parameterization(truss: Truss, limits: Constraints):
    """Use span, horizontal fractions, and heights with geometric upper bounds."""
    # Any path to a support has at most member_max length per edge. Therefore
    # shortest-path hop count * member_max is a necessary bound on joint height.
    neighbors = [set() for _ in truss.nodes]
    for a, b in truss.members:
        neighbors[a].add(b)
        neighbors[b].add(a)
    hops = np.full(len(truss.nodes), np.inf)
    frontier = [truss.pin, truss.roller]
    hops[frontier] = 0
    while frontier:
        following = []
        for node in frontier:
            for neighbor in neighbors[node]:
                if hops[neighbor] > hops[node] + 1:
                    hops[neighbor] = hops[node] + 1
                    following.append(neighbor)
        frontier = following

    free = [i for i in range(len(truss.nodes)) if i not in (truss.pin, truss.roller)]
    bounds = [(limits.span_min, limits.span_max)]
    names = ["support_span_in"]
    x_origin, y_origin = truss.nodes[truss.pin]
    signed_span = truss.nodes[truss.roller, 0] - x_origin
    initial_span = abs(signed_span)
    initial = [initial_span]
    for node in free:
        bounds.append((0.0, 1.0))
        names.append(f"joint_{node + 1}_x_fraction")
        # A right-hand pin is reflected into the canonical left-hand-pin frame.
        # This preserves lengths and vertical-load capacities.
        initial.append((truss.nodes[node, 0] - x_origin) / signed_span)
        height_bounds = (
            (limits.load_height_min, limits.load_height_max)
            if node == truss.load_joint
            else (0.0, float(hops[node] * limits.member_max))
        )
        bounds.append(height_bounds)
        names.append(f"joint_{node + 1}_height_in")
        initial.append(truss.nodes[node, 1] - y_origin)

    def decode(parameters):
        nodes = np.zeros_like(truss.nodes, dtype=float)
        nodes[truss.roller, 0] = parameters[0]
        for index, node in enumerate(free):
            nodes[node] = (parameters[1 + 2 * index] * parameters[0],
                           parameters[2 + 2 * index])
        return Truss(
            name=f"{truss.name} (optimized)", nodes=nodes,
            members=truss.members.copy(), pin=truss.pin, roller=truss.roller,
            load_joint=truss.load_joint,
        )

    return np.asarray(initial), np.asarray(bounds), names, decode


def _objective_value(analysis: Analysis, objective: str) -> float:
    if objective == "load":
        return float(analysis.failure_load)
    if objective == "load-cost":
        return float(analysis.load_cost)
    return float(analysis.lower_failure_load)


def _length_violation(truss: Truss, limits: Constraints) -> float:
    lengths = np.linalg.norm(
        truss.nodes[truss.members[:, 1]] - truss.nodes[truss.members[:, 0]], axis=1
    )
    violations = np.maximum(limits.member_min - lengths, 0)
    violations += np.maximum(lengths - limits.member_max, 0)
    return float(
        np.sum(violations) / limits.member_max
        + max(float(lengths.sum()) - limits.max_total_length, 0)
        / limits.max_total_length
    )


def optimize_geometry(
    truss: Truss,
    *,
    generations: int = 150,
    population: int = 12,
    seed: int = 42,
    objective: str = "load",
    limits: Constraints | None = None,
    buckling: BucklingLaw | None = None,
    callback: Callable[[dict], None] | None = None,
) -> OptimizationResult:
    """Maximize a buckling-based objective without changing member connections.

    Parameters
    ----------
    truss:
        A feasible starting design. A noncompliant equilateral comparison truss
        must first be replaced with a compliant geometry.
    generations:
        Maximum number of differential-evolution generations, excluding the
        initial population. This is an evaluation budget, not a guarantee of
        convergence.
    population:
        Population multiplier: population size is this value times the number
        of geometry parameters (with a minimum of five candidates).
    seed:
        Random seed. A fixed seed and software environment reproduce the run.
    objective:
        ``load``: nominal failure load; ``load-cost``: nominal load divided by
        artificial cost; ``lower-load``: lower buckling-strength envelope.
        The lower envelope is not a calibrated probability of truss survival.
    callback:
        Called once for the initial incumbent and after every generation, with
        the same JSON-friendly record saved in ``history``.

    The search includes support span, all unsupported joint horizontal
    positions, and their heights. Joints are restricted between the supports.
    Height upper bounds follow from the maximum permitted member length and
    graph distance to a support. Geometry and strength checks are hard
    acceptance criteria. Infeasible candidates can guide exploration, but
    cannot replace the best feasible incumbent.
    """
    if objective not in {"load", "load-cost", "lower-load"}:
        raise ValueError("objective must be 'load', 'load-cost', or 'lower-load'")
    if isinstance(generations, bool) or not isinstance(generations, int) or generations < 1:
        raise ValueError("generations must be a positive integer")
    if isinstance(population, bool) or not isinstance(population, int) or population < 1:
        raise ValueError("population must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    limits = limits or Constraints()
    buckling = buckling or BucklingLaw()
    started = perf_counter()
    initial_analysis = solve(truss, buckling=buckling)
    failures = [check for check in validate(truss, initial_analysis, limits) if not check.passed]
    if failures:
        details = "; ".join(f"{check.name}: {check.detail}" for check in failures)
        raise ValueError(f"Optimization needs a feasible starting geometry. {details}")

    initial, bounds, names, decode = _parameterization(truss, limits)
    widths = bounds[:, 1] - bounds[:, 0]
    if not np.isfinite(bounds).all() or np.any(widths < 0):
        raise ValueError("Optimization bounds must be finite, with lower <= upper")
    normalized_initial = np.clip(np.divide(
        initial - bounds[:, 0], widths, out=np.zeros_like(initial), where=widths > 0
    ), 0, 1)
    best_truss = truss
    best_analysis = initial_analysis
    best_value = _objective_value(best_analysis, objective)
    evaluations = 1
    feasible_evaluations = 1
    history: list[dict] = []

    def evaluate(normalized):
        nonlocal best_truss, best_analysis, best_value, evaluations, feasible_evaluations
        evaluations += 1
        try:
            candidate = decode(bounds[:, 0] + normalized * widths)
            length_violation = _length_violation(candidate, limits)
            if length_violation > 1e-10:
                return 1.0 + length_violation
            analysis = solve(candidate, buckling=buckling)
            checks = validate(candidate, analysis, limits)
        except (ValueError, np.linalg.LinAlgError):
            return 1e6
        failures = sum(not check.passed for check in checks)
        if failures:
            return 1.0 + failures
        feasible_evaluations += 1
        value = _objective_value(analysis, objective)
        if value > best_value:
            best_truss, best_analysis, best_value = candidate, analysis, value
        return -value

    def record(generation, convergence):
        entry = {
            "generation": generation,
            "objective": objective,
            "objective_value": best_value,
            "failure_load": float(best_analysis.failure_load),
            "lower_failure_load": float(best_analysis.lower_failure_load),
            "cost": float(best_analysis.cost),
            "load_cost": float(best_analysis.load_cost),
            "nodes": best_truss.nodes.tolist(),
            "evaluations": evaluations,
            "feasible_evaluations": feasible_evaluations,
            "convergence": float(convergence) if np.isfinite(convergence) else None,
        }
        history.append(entry)
        if callback is not None:
            callback(entry.copy())

    record(0, np.nan)
    rng = np.random.default_rng(seed)
    size = max(5, population * len(initial))
    initial_population = rng.uniform(size=(size, len(initial)))
    # Start most candidates near the known feasible geometry, with the rest
    # spanning the full domain. This matters because feasible trusses occupy a
    # small part of the coordinate box; it does not shrink the search bounds.
    local_count = max(1, int(size * 0.75))
    scales = np.geomspace(0.002, 0.08, local_count)[:, None]
    initial_population[:local_count] = np.clip(
        normalized_initial + rng.normal(size=(local_count, len(initial))) * scales,
        0, 1,
    )
    initial_population[0] = normalized_initial

    def generation_finished(x, convergence):
        record(len(history), convergence)
        return False

    result = differential_evolution(
        evaluate, [(0.0, 1.0)] * len(initial), maxiter=generations,
        popsize=population, seed=seed, init=initial_population,
        callback=generation_finished, polish=False, tol=1e-6, atol=1e-8,
        mutation=(0.5, 1.0), recombination=0.7, workers=1,
        updating="immediate",
    )
    # Treat independent validation as the final acceptance boundary too.
    final_checks = validate(best_truss, best_analysis, limits)
    if not all(check.passed for check in final_checks):
        raise RuntimeError("The optimizer's final incumbent failed validation")
    metadata = {
        "algorithm": "scipy.optimize.differential_evolution",
        "buckling_law": asdict(buckling),
        "constraints": asdict(limits),
        "objective": objective,
        "seed": seed,
        "generations_requested": generations,
        "generations_completed": int(result.nit),
        "population_multiplier": population,
        "population_size": size,
        "evaluations": evaluations,
        "solver_nfev": int(result.nfev),
        "feasible_evaluations": feasible_evaluations,
        "elapsed_seconds": perf_counter() - started,
        "converged": bool(result.success),
        "stop_reason": "converged" if result.success else "generation_budget_exhausted",
        "solver_message": str(result.message),
        "initial_objective_value": _objective_value(initial_analysis, objective),
        "best_objective_value": best_value,
        "parameter_bounds": [
            {"name": name, "lower": float(pair[0]), "upper": float(pair[1])}
            for name, pair in zip(names, bounds)
        ],
        "scope": (
            "Fixed connectivity and designated support/load joints; geometry only. "
            "All joints lie between the supports. Best feasible design found, "
            "not a proven global optimum."
        ),
        "coordinate_frame": "Candidate pin at (0, 0), roller to the right; input may be reflected",
        "initialization": "75% local perturbations; 25% uniform domain; includes baseline",
    }
    return OptimizationResult(best_truss, best_analysis, history, metadata)
