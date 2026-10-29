"""JSON inputs, report tables, and the course's MATLAB exchange format."""

from __future__ import annotations

import csv
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
from scipy.io import savemat

from .analysis import Analysis, matlab_arrays
from .constraints import stock_cut_plan, validate
from .model import BucklingLaw, Constraints, Truss


def write_json(path: str | Path, data) -> Path:
    """Write portable JSON; undefined numeric values become null, never NaN."""
    def clean(value):
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(data), indent=2, allow_nan=False) + "\n")
    return path


def read_truss(path: str | Path) -> Truss:
    return Truss.from_dict(json.loads(Path(path).read_text()))


def write_truss(path: str | Path, truss: Truss) -> Path:
    return write_json(path, truss.to_dict())


def write_csv(path: Path, rows: list[dict]) -> Path:
    if not rows:
        return path
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def environment() -> dict:
    return {
        "python": platform.python_version(),
        **{package: version(package) for package in ("numpy", "scipy", "matplotlib", "pillow")},
    }


def member_rows(a: Analysis, buckling: BucklingLaw) -> list[dict]:
    rows = []
    for i, (start, end) in enumerate(a.truss.members):
        ratio = float(a.unit_forces[i])
        state = "C" if ratio < -1e-9 else "T" if ratio > 1e-9 else "Z"
        compression = state == "C"
        rows.append({
            "member": f"M{i + 1}",
            "start_joint": f"J{start + 1}",
            "end_joint": f"J{end + 1}",
            "length_in": float(a.lengths[i]),
            "state": state,
            "force_at_applied_load_oz": float(abs(a.forces[i])),
            "signed_force_at_applied_load_oz": float(a.forces[i]),
            "force_at_nominal_failure_oz": float(abs(ratio) * a.failure_load),
            "buckling_capacity_oz": float(a.capacities[i]) if compression else "",
            "buckling_uncertainty_oz": buckling.uncertainty_oz if compression else "",
            "applied_load_to_buckle_oz": float(a.capacities[i] / -ratio) if compression else "",
            "critical": bool(i in a.critical_members),
        })
    return rows


def export_analysis(
    a: Analysis,
    output_dir: str | Path,
    *,
    buckling: BucklingLaw | None = None,
    limits: Constraints = Constraints(),
    team: str = "Not specified",
    section: str = "Not specified",
) -> dict:
    """Export auditable results; invalid designs are explicitly marked, not hidden."""
    buckling = buckling or a.buckling
    if buckling != a.buckling:
        raise ValueError("Export buckling law must match the law used for analysis")
    if not np.isfinite(a.failure_load):
        raise ValueError("No finite buckling prediction: this load case has no compressed members")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checks = validate(a.truss, a, limits=limits)
    rows = member_rows(a, buckling)
    summary = {
        "name": a.truss.name,
        "team": team,
        "section": section,
        "units": {"length": "in", "force": "oz-force", "cost": "course currency"},
        "applied_load_oz": a.load_oz,
        "nominal_failure_load_oz": a.failure_load,
        "critical_member_uncertainty_oz": a.uncertainty,
        "all_member_lower_envelope_oz": a.lower_failure_load,
        "all_member_upper_envelope_oz": a.upper_failure_load,
        "critical_members": [f"M{i + 1}" for i in a.critical_members],
        "cost": a.cost,
        "load_cost_oz_per_currency": a.load_cost,
        "total_member_length_in": float(np.sum(a.lengths)),
        "equilibrium_residual_oz": a.residual,
        "matrix_condition_number": a.condition,
        "passes_all_checks": all(check.passed for check in checks),
        "checks": [asdict(check) for check in checks],
        "buckling_law": asdict(buckling),
        "constraints": asdict(limits),
        "environment": environment(),
        "limitations": [
            "Static, planar, pin-jointed model; no physical 60-second test is simulated.",
            "Tension rupture, joint failure, self-weight, and out-of-plane instability are excluded.",
            "Fit error is not a probability model for independent member strengths.",
        ],
    }
    write_json(out / "summary.json", summary)
    write_truss(out / "geometry.json", a.truss)
    write_csv(out / "members.csv", rows)
    write_csv(out / "joints.csv", [
        {"joint": f"J{i+1}", "x_in": float(x), "y_in": float(y)}
        for i, (x, y) in enumerate(a.truss.nodes)
    ])
    write_csv(out / "reactions.csv", [
        {"reaction": label, "signed_force_oz": float(value)}
        for label, value in zip(("pin_x", "pin_y", "roller_y"), a.reactions)
    ])
    savemat(out / "design.mat", matlab_arrays(a.truss, a.load_oz), do_compression=True)
    plan = stock_cut_plan(a.lengths, limits.stock_lengths, limits.kerf_in)
    if plan is not None:
        write_csv(out / "cut_plan.csv", [
            {"stock_bar": bar + 1, "stock_length_in": limits.stock_lengths[bar],
             "member": f"M{member+1}", "length_in": float(a.lengths[member])}
            for bar, members in enumerate(plan) for member in members
        ])
    else:
        (out / "cut_plan.csv").unlink(missing_ok=True)

    lines = [
        f"# {a.truss.name}", "", f"Team: {team}; section: {section}.", "",
        f"**Constraint status: {'PASS' if summary['passes_all_checks'] else 'FAIL'}.**",
        "", f"Applied load: {a.load_oz:.3f} oz-force downward.",
        f"Nominal failure load: {a.failure_load:.3f} oz-force; "
        f"critical-member propagated uncertainty: ±{a.uncertainty:.3f} oz-force.",
        f"All-member fit envelopes: [{a.lower_failure_load:.3f}, {a.upper_failure_load:.3f}] oz-force.",
        f"First buckling member(s): {', '.join(summary['critical_members'])}.",
        f"Cost: {a.cost:.3f}; maximum-load/cost: {a.load_cost:.5f} oz-force/currency.",
        "", "Positive signed axial force means tension; negative means compression. Z means zero force.",
        "", "| Member | Length (in) | State | Force at applied load (oz) | Force at failure (oz) | Buckling capacity ± fit error (oz) |",
        "|---|---:|:---:|---:|---:|---:|",
    ]
    tex = [r"\begin{tabular}{lrrrrr}", r"\hline",
           r"Member & Length (in) & T/C/Z & $|F|$ (oz) & $|F_{\max}|$ (oz) & $F_b \pm U$ (oz) \\", r"\hline"]
    for row in rows:
        capacity = (f"{row['buckling_capacity_oz']:.3f} ± {buckling.uncertainty_oz:.2f}"
                    if row["state"] == "C" else "—")
        fields = [row["member"], f"{row['length_in']:.3f}", row["state"],
                  f"{row['force_at_applied_load_oz']:.3f}",
                  f"{row['force_at_nominal_failure_oz']:.3f}", capacity]
        lines.append("| " + " | ".join(fields) + " |")
        tex.append(" & ".join(fields).replace("±", r"$\pm$").replace("—", "--") + r" \\")
    tex.extend([r"\hline", r"\end{tabular}"])
    lines.extend(["", "## Constraint checks", ""])
    lines.extend(f"- {'PASS' if c.passed else 'FAIL'}: {c.name}: {c.detail}" for c in checks)
    lines.extend(["", "## Scope", ""] + summary["limitations"])
    lines.extend(["", "## Figure captions", "",
                  "- Geometry: joint and member identifiers with dimensions in inches.",
                  "- Member forces: axial force magnitudes, tension/compression, supports, and external loads.",
                  "- Buckling: compression demand and the empirical member strength with fit uncertainty.",
                  "- Load response: linear force predictions up to the first predicted buckle.",
                  "- Buckling fit: the supplied empirical fit and error band, not new experimental data."])
    (out / "analysis.md").write_text("\n".join(lines) + "\n")
    (out / "members.tex").write_text("\n".join(tex) + "\n")
    return summary
