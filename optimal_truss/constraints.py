"""Geometry, course-limit, and material-cutting checks with explicit explanations."""

from dataclasses import dataclass

import numpy as np

from .analysis import equilibrium_matrix
from .model import Constraints

TOL = 1e-7


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def stock_cut_plan(lengths, stock_lengths=(48.0, 48.0, 24.0), kerf_in=0.0):
    """Assign every member to a stock bar, or return None if none fits.

    Entries are zero-based member indexes in each stock bar. A conservative
    kerf is charged once per piece, including the final piece of a bar. This
    models joint-to-joint lengths only; joint overlap allowances are separate.
    """
    lengths = np.asarray(lengths, dtype=float)
    stock = np.asarray(stock_lengths, dtype=float)
    if (not np.isfinite(lengths).all() or np.any(lengths <= 0)
            or not np.isfinite(stock).all() or np.any(stock <= 0)
            or not np.isfinite(kerf_in) or kerf_in < 0):
        raise ValueError("Lengths/stock must be positive and finite; kerf must be nonnegative")
    pieces = lengths + kerf_in
    if pieces.sum() > stock.sum() + TOL:
        return None
    order = np.argsort(-pieces)
    remaining, plan = stock.copy(), [[] for _ in stock]

    def place(k):
        if k == len(order):
            return True
        member = int(order[k])
        seen = []
        for bar in np.argsort(remaining):
            room = remaining[bar]
            if any(abs(room - prior) < TOL for prior in seen):
                continue
            seen.append(room)
            if pieces[member] <= room + TOL:
                remaining[bar] -= pieces[member]
                plan[bar].append(member)
                if place(k + 1):
                    return True
                plan[bar].pop()
                remaining[bar] += pieces[member]
        return False

    return plan if place(0) else None


def _simple_triangles(truss):
    adjacency = {i: set() for i in range(len(truss.nodes))}
    for i, j in truss.members:
        adjacency[i].add(j)
        adjacency[j].add(i)
    while len(adjacency) > 3:
        for node, neighbors in adjacency.items():
            if len(neighbors) == 2:
                i, j = neighbors
                if j in adjacency[i]:
                    adjacency[i].remove(node)
                    adjacency[j].remove(node)
                    del adjacency[node]
                    break
        else:
            return False
    return len(adjacency) == 3 and all(len(v) == 2 for v in adjacency.values())


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _on_segment(point, a, b):
    return (abs(_cross(b - a, point - a)) <= TOL
            and np.dot(point - a, point - b) <= TOL)


def _geometry_issues(truss):
    issues = []
    for k, (i, j) in enumerate(truss.members):
        a, b = truss.nodes[[i, j]]
        for p, point in enumerate(truss.nodes):
            if p not in (i, j) and _on_segment(point, a, b):
                issues.append(f"M{k + 1} passes through J{p + 1}")
        for q in range(k):
            u, v = truss.members[q]
            if len({int(i), int(j), int(u), int(v)}) < 4:
                continue  # Shared-endpoint overlaps are caught by the through-joint check.
            c, d = truss.nodes[[u, v]]
            crosses = (_cross(b - a, c - a) * _cross(b - a, d - a) < -TOL
                       and _cross(d - c, a - c) * _cross(d - c, b - c) < -TOL)
            touches = any((_on_segment(c, a, b), _on_segment(d, a, b),
                           _on_segment(a, c, d), _on_segment(b, c, d)))
            if crosses or touches:
                issues.append(f"M{k + 1} intersects M{q + 1}")
    return issues


def validate(truss, analysis=None, limits=Constraints()):
    """Return every check, including failures; geometry alone does not certify load.

    Passing the nominal 30 oz prediction does not verify physical hold time.
    Supplying an Analysis avoids repeating its matrix stability calculation.
    """
    nodes, members = truss.nodes, truss.members
    lengths = np.linalg.norm(nodes[members[:, 1]] - nodes[members[:, 0]], axis=1)
    pin, roller, loaded = nodes[[truss.pin, truss.roller, truss.load_joint]]
    span, height = abs(roller[0] - pin[0]), loaded[1] - pin[1]
    checks = []

    def add(name, passed, detail):
        checks.append(Check(name, bool(passed), detail))

    add("member_count", len(members) == 2 * len(nodes) - 3,
        f"M={len(members)}, J={len(nodes)}; required M=2J-3")
    add("simple_triangles", _simple_triangles(truss),
        "Connectivity must be constructible from a triangle by adding a joint and two members")
    issues = _geometry_issues(truss)
    add("planar_members", not issues, "; ".join(issues) if issues else "No crossings, overlaps, or members through other joints")
    add("positive_lengths", np.all(lengths > TOL), "Every member must have nonzero length")
    stable = analysis is not None and np.isfinite(analysis.condition) and analysis.condition <= 1e12
    if analysis is None:
        try:
            a = equilibrium_matrix(truss)
            stable = a.shape[0] == a.shape[1] and np.linalg.cond(a) <= 1e12
        except ValueError:
            stable = False
    add("stable", stable, "Equilibrium matrix must be square and nonsingular (condition <= 1e12)")
    add("support_baseline", abs(pin[1] - roller[1]) <= TOL,
        "Pin and roller must lie at the same height")
    add("support_ends", abs(min(pin[0], roller[0]) - nodes[:, 0].min()) <= TOL
        and abs(max(pin[0], roller[0]) - nodes[:, 0].max()) <= TOL,
        "Supports must be the leftmost and rightmost joints")
    add("above_supports", np.all(nodes[:, 1] >= pin[1] - TOL), "No member or joint may extend below the support line")
    add("span", limits.span_min - TOL <= span <= limits.span_max + TOL,
        f"Span {span:.6g} in; allowed {limits.span_min:g}-{limits.span_max:g} in")
    add("member_lengths", np.all(lengths >= limits.member_min - TOL)
        and np.all(lengths <= limits.member_max + TOL),
        f"Member lengths {lengths.min():.6g}-{lengths.max():.6g} in; allowed {limits.member_min:g}-{limits.member_max:g} in")
    add("load_height", limits.load_height_min - TOL <= height <= limits.load_height_max + TOL,
        f"Load joint height {height:.6g} in; allowed {limits.load_height_min:g}-{limits.load_height_max:g} in")
    under = (np.abs(nodes[:, 0] - loaded[0]) <= TOL) & (nodes[:, 1] < loaded[1] - TOL)
    add("load_clearance", not np.any(under), "No joint may be directly below the load joint")
    add("load_off_supports", truss.load_joint not in (truss.pin, truss.roller), "Load must act at an unsupported joint")
    add("total_material", lengths.sum() <= limits.max_total_length + TOL,
        f"Total joint-to-joint length {lengths.sum():.6g} in; limit {limits.max_total_length:g} in")
    plan = stock_cut_plan(lengths, limits.stock_lengths, limits.kerf_in) if np.all(lengths > 0) else None
    add("stock_cutting", plan is not None,
        f"Pieces must fit stock {limits.stock_lengths} in, with {limits.kerf_in:g} in kerf per piece; joint allowances excluded")
    if analysis is not None:
        add("minimum_load", np.isfinite(analysis.failure_load) and analysis.failure_load >= limits.min_load - TOL,
            f"Predicted nominal capacity {analysis.failure_load:.6g} oz; target {limits.min_load:g} oz (hold time unverified)")
    return checks
