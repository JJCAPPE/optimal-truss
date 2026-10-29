"""Report figures in inches and ounce-force, plus recorded-search animations.

Every static figure is saved as a 240 dpi PNG and editable vector SVG/PDF.
Functions return the written paths and never open a GUI. Joint and member IDs
are one-based in figures, matching the exported report tables.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
import numpy as np

from .analysis import Analysis, solve
from .model import BucklingLaw, Truss


TENSION = "#0072B2"
COMPRESSION = "#D55E00"
NEUTRAL = "#777777"
INK = "#17212B"
ACCENT = "#009E73"
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 15,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#A8B0B5",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
}


def _save(fig, output_dir: str | Path, name: str) -> list[Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    try:
        for suffix in ("png", "svg", "pdf"):
            path = directory / f"{name}.{suffix}"
            fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.18)
            paths.append(path)
    finally:
        plt.close(fig)
    return paths


def _force_style(force: float) -> tuple[str, str, str]:
    if abs(force) < 1e-8:
        return NEUTRAL, ":", "0"
    return (TENSION, "-", "T") if force > 0 else (COMPRESSION, "--", "C")


def _analysis_law(result: Analysis, buckling: BucklingLaw | None) -> BucklingLaw:
    if buckling is not None and buckling != result.buckling:
        raise ValueError("The plotting buckling law must match the law used for the analysis")
    return result.buckling


def _require_finite_failure(result: Analysis) -> None:
    if not np.isfinite(result.failure_load) or not np.any(result.unit_forces < -1e-10):
        raise ValueError("Buckling report figures require compressed members and a finite failure load")


def _support(ax, x: float, y: float, roller: bool, scale: float) -> None:
    width, height = scale * 0.65, scale * 0.85
    triangle = Polygon(
        [(x, y), (x - width, y - height), (x + width, y - height)],
        closed=True, facecolor="white", edgecolor=INK, linewidth=1.2, zorder=6,
    )
    ax.add_patch(triangle)
    base = y - height
    if roller:
        radius = scale * 0.15
        for dx in (-scale * 0.35, scale * 0.35):
            ax.add_patch(Circle((x + dx, base - radius), radius,
                                facecolor="white", edgecolor=INK, linewidth=1))
        base -= 2 * radius
    ax.plot([x - width * 1.3, x + width * 1.3], [base, base],
            color=INK, linewidth=1)
    for dx in np.linspace(-width, width, 5):
        ax.plot([x + dx, x + dx - scale * 0.25], [base, base - scale * 0.3],
                color=NEUTRAL, linewidth=0.7)


def _arrow(ax, start, end, text, *, color=INK, offset=(7, 0), ha="left"):
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.7,
                            "mutation_scale": 13}, zorder=8)
    midpoint = (np.asarray(start) + np.asarray(end)) / 2
    ax.annotate(text, midpoint, xytext=offset, textcoords="offset points",
                ha=ha, va="center", color=color, fontsize=9,
                bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none", "pad": 1.5},
                zorder=9)


def _draw_structure(ax, result: Analysis, mode: str = "forces", *,
                    bounds=None, title: str | None = None) -> None:
    truss = result.truss
    nodes = np.asarray(truss.nodes)
    span = float(np.ptp(nodes[:, 0]))
    scale = max(span / 35, 0.4)
    critical = set(result.critical_members)
    for m, (i, j) in enumerate(truss.members):
        a, b = nodes[i], nodes[j]
        if mode == "forces":
            color, style, state = _force_style(result.forces[m])
        else:
            color, style, state = INK, "-", ""
        if mode == "forces" and m in critical:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#E0B449",
                    linewidth=7, alpha=.55, solid_capstyle="round", zorder=1)
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color,
                linestyle=style, linewidth=2, solid_capstyle="round", zorder=2)
        if mode != "fbd":
            midpoint = (a + b) / 2
            d = b - a
            normal = np.array([-d[1], d[0]]) / np.linalg.norm(d)
            if normal[1] < 0:
                normal *= -1
            if mode == "geometry":
                label = f"M{m + 1} · {result.lengths[m]:.2f} in"
            else:
                label = f"M{m + 1} · {abs(result.forces[m]):.2f} {state}"
            ax.annotate(label, midpoint, xytext=tuple(9 * normal),
                        textcoords="offset points", ha="center", va="center",
                        fontsize=8.3, color=color,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .92, "pad": 1.5},
                        zorder=7)
    ax.scatter(nodes[:, 0], nodes[:, 1], s=33, color="white", edgecolors=INK,
               linewidths=1.2, zorder=8)
    for j, (x, y) in enumerate(nodes):
        offset = (0, 12) if y > np.median(nodes[:, 1]) + .01 else (0, -14)
        if j in (truss.pin, truss.roller):
            offset = (-15, 8) if j == truss.pin else (15, 8)
        if j == truss.load_joint:
            offset = (-12, 8)
        ax.annotate(f"J{j + 1}", (x, y), xytext=offset, textcoords="offset points",
                    ha="center", va="center", fontsize=9, fontweight="bold", zorder=9)
    for j, roller in ((truss.pin, False), (truss.roller, True)):
        _support(ax, *nodes[j], roller, scale)
    if mode != "geometry":
        p = nodes[truss.load_joint]
        _arrow(ax, p, p + (0, -3 * scale), f"W = {result.load_oz:.2f} oz",
               color=INK)
        for j, reaction, label, side in (
            (truss.pin, result.reactions[1], "Ry,p", -1),
            (truss.roller, result.reactions[2], "Ry,r", 1),
        ):
            p = nodes[j]
            direction = 1 if reaction >= 0 else -1
            _arrow(ax, p, p + (0, direction * 2.8 * scale),
                   f"{label} = {reaction:.2f} oz", color=ACCENT,
                   offset=(side * 8, 0), ha="right" if side < 0 else "left")
        if abs(result.reactions[0]) > 1e-8:
            p = nodes[truss.pin]
            _arrow(ax, p, p + (np.sign(result.reactions[0]) * 2.8 * scale, 0),
                   f"Rx,p = {result.reactions[0]:.2f} oz", color=ACCENT,
                   offset=(0, 12), ha="center")
    if bounds is None:
        bounds = (nodes[:, 0].min() - 5 * scale, nodes[:, 0].max() + 5 * scale,
                  min(nodes[:, 1].min() - 4.3 * scale, -4.3 * scale),
                  nodes[:, 1].max() + 2.5 * scale)
    ax.set(xlim=bounds[:2], ylim=bounds[2:], xlabel="x (in)", ylabel="y (in)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title or truss.name, loc="left", pad=18)
    ax.grid(axis="both", color="#EEF0F2", linewidth=.7)
    ax.set_axisbelow(True)


def plot_truss(result: Analysis, output_dir: str | Path, *,
               mode: str = "forces", name: str | None = None) -> list[Path]:
    """Draw geometry, signed member forces, or an external free-body diagram.

    ``mode`` is ``geometry``, ``forces`` or ``fbd``. Force annotations use ounce-
    force; T/C means tension/compression. Highlighting identifies the member(s)
    predicted to buckle first, even when the displayed load is below failure.
    """
    if mode not in {"geometry", "forces", "fbd"}:
        raise ValueError("mode must be geometry, forces, or fbd")
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(12, 6.2))
        fig.subplots_adjust(bottom=.24, top=.85)
        _draw_structure(ax, result, mode)
        if mode == "forces":
            handles = [Line2D([], [], color=TENSION, lw=2, label="Tension (T)"),
                       Line2D([], [], color=COMPRESSION, lw=2, ls="--", label="Compression (C)"),
                       Line2D([], [], color=NEUTRAL, lw=2, ls=":", label="Zero force"),
                       Line2D([], [], color="#E0B449", lw=6, alpha=.55, label="First buckling")]
            fig.legend(handles=handles, loc="lower center", ncol=4,
                       frameon=False, bbox_to_anchor=(.5, .07))
            note = (f"Member forces: oz-force. Predicted failure: {result.failure_load:.2f} "
                    f"± {result.uncertainty:.2f} oz.  Rx,p = {result.reactions[0]:.2f} oz.")
        elif mode == "geometry":
            note = (f"Dimensions: inches. {len(result.truss.nodes)} joints; "
                    f"{len(result.truss.members)} members; total length {sum(result.lengths):.2f} in.")
        else:
            note = "External forces only. Internal member forces cancel for the complete truss."
        fig.text(.5, .025, note, ha="center", fontsize=9, color=NEUTRAL)
        return _save(fig, output_dir, name or mode)


def plot_joint(result: Analysis, joint: int, output_dir: str | Path) -> list[Path]:
    """Draw actual force directions on a single zero-based joint index."""
    truss = result.truss
    if not 0 <= joint < len(truss.nodes):
        raise ValueError("joint index is outside the truss")
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8.5, 7))
        origin = np.zeros(2)
        components = []
        for m, (i, j) in enumerate(truss.members):
            if joint not in (i, j):
                continue
            other = j if joint == i else i
            direction = (truss.nodes[other] - truss.nodes[joint]) / result.lengths[m]
            force = result.forces[m]
            color, _, state = _force_style(force)
            vector = direction * force
            components.append(vector)
            if abs(force) < 1e-8:
                continue
            end = direction * np.sign(force) * 1.8
            ax.annotate("", xy=end, xytext=origin,
                        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2, "mutation_scale": 15})
            pos = end * 1.18
            ax.text(*pos, f"M{m + 1}\n{abs(force):.2f} oz ({state})",
                    color=color, ha="center", va="center", fontsize=10)
        external = np.zeros(2)
        if joint == truss.load_joint:
            external[1] -= result.load_oz
        if joint == truss.pin:
            external += result.reactions[:2]
        if joint == truss.roller:
            external[1] += result.reactions[2]
        for axis, component in enumerate(external):
            if abs(component) > 1e-8:
                end = np.zeros(2)
                end[axis] = 1.8 * np.sign(component)
                ax.annotate("", xy=end, xytext=origin,
                            arrowprops={"arrowstyle": "-|>", "color": ACCENT,
                                        "lw": 2, "mutation_scale": 15})
                ax.text(*(end * 1.2), f"External\n{abs(component):.2f} oz",
                        color=ACCENT, ha="center", va="center")
        residual = np.sum(components, axis=0) + external
        ax.scatter([0], [0], s=110, facecolor="white", edgecolor=INK, zorder=5)
        ax.text(0, .18, f"J{joint + 1}", ha="center", fontweight="bold")
        ax.set(xlim=(-2.8, 2.8), ylim=(-2.8, 2.8), aspect="equal")
        ax.set_title(f"Joint J{joint + 1} · {truss.name}", loc="left", pad=15)
        ax.axis("off")
        fig.text(.5, .04, "Arrow lengths are schematic; directions and force values are physical.\n"
                 f"ΣFx = {residual[0]:.2e} oz;  ΣFy = {residual[1]:.2e} oz.",
                 ha="center", fontsize=9)
        return _save(fig, output_dir, f"joint_J{joint + 1}")


def plot_buckling(result: Analysis, output_dir: str | Path,
                  buckling: BucklingLaw | None = None) -> list[Path]:
    """Compare compressed-member demand with fitted capacity and fit uncertainty."""
    buckling = _analysis_law(result, buckling)
    _require_finite_failure(result)
    compressed = np.flatnonzero(result.unit_forces < -1e-10)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        positions = np.arange(len(compressed))
        demand = -result.forces[compressed]
        capacity = result.capacities[compressed]
        ax.bar(positions - .19, demand, .36, label=f"Demand at W = {result.load_oz:.2f} oz",
               color=COMPRESSION, alpha=.86)
        ax.bar(positions + .19, capacity, .36, yerr=buckling.uncertainty_oz,
               capsize=4, label="Buckling capacity ± fit uncertainty", color=TENSION, alpha=.82)
        ax.set_xticks(positions, [f"M{i + 1}" + (" *" if i in result.critical_members else "")
                                 for i in compressed])
        ax.set(xlabel="Compressed member (* first buckling)", ylabel="Force (oz)",
               title=f"Compression demand and capacity · {result.truss.name}")
        ax.set_ylim(0, max(float(np.max(demand)),
                           float(np.max(capacity)) + buckling.uncertainty_oz) * 1.35)
        ax.grid(axis="y", color="#E7EBEE")
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()
        return _save(fig, output_dir, "buckling_capacity")


def plot_buckling_fit(output_dir: str | Path, buckling: BucklingLaw = BucklingLaw(),
                     *, lengths: tuple[float, float] = (8.5, 14)) -> list[Path]:
    """Plot the supplied class fit, not fabricated measurements or scatter."""
    x = np.linspace(*lengths, 250)
    y = buckling.capacity(x)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax.fill_between(x, y - buckling.uncertainty_oz, y + buckling.uncertainty_oz,
                        alpha=.18, color=TENSION,
                        label=f"Fit uncertainty ±{buckling.uncertainty_oz:g} oz")
        ax.plot(x, y, color=TENSION, lw=2.2,
                label=f"Fb = {buckling.coefficient:.9g} L$^{{-{buckling.exponent:g}}}$")
        ax.set(xlabel="Member length L (in)", ylabel="Buckling force Fb (oz)",
               title="Class buckling relationship", xlim=lengths)
        ax.grid(color="#E7EBEE", lw=.7)
        ax.legend(frameon=False)
        fig.tight_layout()
        return _save(fig, output_dir, "buckling_fit")


def plot_load_response(result: Analysis, output_dir: str | Path) -> list[Path]:
    """Show linear member force scaling up to first nominal buckling only."""
    _require_finite_failure(result)
    loads = np.linspace(0, result.failure_load, 100)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#333333", "#56B4E9", "#9A6300"]
        markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "p", "+", "x"]
        groups = []
        for m, ratio in enumerate(result.unit_forces):
            if abs(ratio) < 1e-8:
                continue
            matching = next((g for g in groups if np.isclose(g[0], ratio, atol=1e-9, rtol=1e-9)), None)
            if matching is None:
                groups.append((ratio, [m]))
            else:
                matching[1].append(m)
        for k, (ratio, members) in enumerate(groups):
            label = "/".join(f"M{m + 1}" for m in members)
            if any(m in result.critical_members for m in members):
                label += " *"
            ax.plot(loads, ratio * loads, color=colors[k % len(colors)],
                    ls="-" if ratio > 0 else "--", lw=1.6,
                    marker=markers[k % len(markers)], markevery=20, markersize=3.5,
                    markerfacecolor="white", label=label)
        ax.axhline(0, color=NEUTRAL, lw=.8)
        ax.axvline(result.failure_load, color=INK, ls=":", lw=1.5)
        ax.set(xlabel="Applied load W (oz)", ylabel="Member force (oz): + tension / − compression",
               title=f"Load response until first buckling · {result.truss.name}",
               xlim=(0, result.failure_load * 1.04))
        ax.grid(color="#E7EBEE", lw=.7)
        ax.legend(title="* First buckling", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
        fig.tight_layout()
        return _save(fig, output_dir, "load_response")


def plot_comparison(analyses: Sequence[Analysis], output_dir: str | Path) -> list[Path]:
    """Compare nominal capacities, artificial cost and load/cost without mixed axes."""
    if not analyses:
        raise ValueError("at least one analysis is required")
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(13, 5.4))
        x = np.arange(len(analyses))
        labels = [a.truss.name.replace(" ", "\n", 2) for a in analyses]
        panels = [([a.failure_load for a in analyses], "Maximum load (oz)",
                   [a.uncertainty for a in analyses]),
                  ([a.cost for a in analyses], "Artificial cost ($301)", None),
                  ([a.load_cost for a in analyses], "Maximum load / cost (oz/$301)", None)]
        colors = [TENSION if i % 2 == 0 else ACCENT for i in x]
        for ax, (values, ylabel, error) in zip(axes, panels):
            ax.bar(x, values, color=colors, width=.6, yerr=error, capsize=5)
            for i, value in enumerate(values):
                ax.annotate(f"{value:.2f}", (i, value + (error[i] if error else 0)),
                            xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
            ax.set_xticks(x, labels, fontsize=9)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, max(v + (error[i] if error else 0) for i, v in enumerate(values)) * 1.27)
            ax.grid(axis="y", color="#E7EBEE")
            ax.set_axisbelow(True)
        axes[0].axhline(30, color=INK, lw=1, ls=":", label="30 oz target")
        axes[0].legend(frameon=False, fontsize=9)
        fig.suptitle("Design comparison", x=.07, ha="left", fontsize=17)
        fig.tight_layout(rect=(0, 0, 1, .94))
        return _save(fig, output_dir, "design_comparison")


def report_figures(result: Analysis, output_dir: str | Path,
                   buckling: BucklingLaw | None = None) -> list[Path]:
    """Generate the report set using the original analysis's buckling law.

    An explicitly supplied law must match the analyzed law. A report requiring
    first-buckling figures needs a finite failure load; this is checked before
    writing any files. Use ``plot_truss`` for an equilibrium-only diagram.
    """
    buckling = _analysis_law(result, buckling)
    _require_finite_failure(result)
    paths = []
    for mode in ("geometry", "forces", "fbd"):
        paths += plot_truss(result, output_dir, mode=mode)
    at_failure = solve(result.truss, load_oz=result.failure_load, buckling=buckling)
    paths += plot_truss(at_failure, output_dir, mode="forces", name="forces_at_failure")
    paths += plot_joint(result, result.truss.load_joint, output_dir)
    paths += plot_buckling(result, output_dir, buckling)
    paths += plot_buckling_fit(output_dir, buckling)
    paths += plot_load_response(result, output_dir)
    return paths


def plot_optimization(history: Sequence[dict], base_truss: Truss, output_dir: str | Path,
                      buckling: BucklingLaw = BucklingLaw(), *, objective: str = "load",
                      fps: int = 6, max_frames: int = 80) -> list[Path]:
    """Plot recorded generations and animate their actual feasible geometries.

    History follows ``OptimizationResult.history``: generation, nodes,
    failure_load, cost, objective_value. Repeated unchanged incumbents remain
    genuine records. When subsampling, the first and last records are retained.
    No tweened geometry or unrecorded optimizer evaluations are invented.
    """
    records = [row for row in history if row.get("feasible", True)]
    if not records:
        raise ValueError("optimization history contains no feasible records")
    if fps < 1 or max_frames < 2:
        raise ValueError("fps must be positive and max_frames at least 2")
    objective = records[-1].get("objective", objective)
    units = "oz/$301" if objective == "load-cost" else "oz"
    generations = [r["generation"] for r in records]
    scores = [r.get("objective_value", r.get("score", r["failure_load"])) for r in records]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.2), sharex=True)
        axes[0].step(generations, scores, where="post", color=TENSION, lw=2)
        axes[0].set(ylabel=f"Best feasible objective ({units})",
                    title=f"Optimization progress · {objective}")
        axes[1].plot(generations, [r["failure_load"] for r in records], color=ACCENT,
                     lw=1.8, label="Nominal maximum load")
        if all("lower_failure_load" in r for r in records):
            axes[1].plot(generations, [r["lower_failure_load"] for r in records],
                         color=COMPRESSION, ls="--", label="Lower capacity envelope")
        axes[1].axhline(30, color=NEUTRAL, ls=":", lw=1, label="30 oz target")
        axes[1].set(xlabel="Recorded generation (0 = starting design)", ylabel="Supported load (oz)")
        axes[1].legend(frameon=False)
        for ax in axes:
            ax.grid(color="#E7EBEE", lw=.7)
        fig.tight_layout()
        paths = _save(fig, output_dir, "optimization_history")
        indices = np.unique(np.linspace(0, len(records) - 1, min(max_frames, len(records)), dtype=int))
        frames = [records[i] for i in indices]
        all_nodes = np.concatenate([np.asarray(r["nodes"]) for r in frames])
        span = float(np.ptp(all_nodes[:, 0]))
        scale = max(span / 35, .4)
        bounds = (all_nodes[:, 0].min() - 5 * scale, all_nodes[:, 0].max() + 5 * scale,
                  all_nodes[:, 1].min() - 4.3 * scale, all_nodes[:, 1].max() + 2.5 * scale)
        fig, ax = plt.subplots(figsize=(11, 6))
        fig.subplots_adjust(bottom=.17, top=.88)
        sampling = (f"All {len(records)} recorded feasible states."
                    if len(frames) == len(records)
                    else f"{len(frames)} of {len(records)} recorded feasible states; evenly sampled, endpoints retained.")
        fig.text(.5, .035, sampling + "\nForces shown at predicted nominal failure load.\n"
                 "Blue solid: tension; orange dashed: compression; gold highlight: first buckling.",
                 ha="center", fontsize=9)

        def draw(frame):
            ax.clear()
            truss = replace(base_truss, nodes=np.asarray(frame["nodes"], dtype=float))
            result = solve(truss, load_oz=frame["failure_load"], buckling=buckling)
            title = (f"Generation {frame['generation']}  |  Capacity {result.failure_load:.2f} oz"
                     f"  |  Cost {result.cost:.2f} $301")
            _draw_structure(ax, result, bounds=bounds, title=title)
            return []

        movie = animation.FuncAnimation(fig, draw, frames=frames, interval=1000 / fps, blit=False)
        path = Path(output_dir) / "optimization.gif"
        try:
            movie.save(path, writer=animation.PillowWriter(fps=fps), dpi=110)
        finally:
            plt.close(fig)
        paths.append(path)
        return paths
