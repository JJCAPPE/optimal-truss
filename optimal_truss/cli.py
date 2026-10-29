"""Command-line workflow. Run `python -m optimal_truss --help`."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

from .analysis import solve
from .geometry import warren
from .io import environment, export_analysis, read_truss, write_csv, write_json, write_truss
from .model import Constraints
from .optimize import optimize_geometry
from .plotting import plot_comparison, plot_optimization, report_figures


def _result(a, out, args):
    summary = export_analysis(a, out, limits=Constraints(kerf_in=args.kerf),
                              team=args.team, section=args.section)
    if not args.no_figures:
        report_figures(a, Path(out) / "figures")
    status = "PASS" if summary["passes_all_checks"] else "FAIL"
    print(f"{a.truss.name}: {a.failure_load:.3f} ± {a.uncertainty:.3f} oz; "
          f"cost {a.cost:.3f}; load/cost {a.load_cost:.5f}; constraints {status}")
    print(f"  Results: {Path(out).resolve()}")
    return summary


def _optimize(truss, out: Path, args):
    results = []
    for seed in args.seeds:
        def progress(record):
            if record["generation"] % 10 == 0:
                print(f"  Seed {seed}, generation {record['generation']}: "
                      f"best feasible load {record['failure_load']:.3f} oz", flush=True)

        result = optimize_geometry(truss, generations=args.generations,
                                   population=args.population, seed=seed,
                                   objective=args.objective,
                                   limits=Constraints(kerf_in=args.kerf),
                                   callback=progress)
        run_dir = out / f"seed-{seed}"
        write_json(run_dir / "history.json", result.history)
        write_json(run_dir / "metadata.json", result.metadata)
        write_truss(run_dir / "geometry.json", result.truss)
        results.append(result)
        print(f"  Seed {seed} finished: {result.analysis.failure_load:.3f} oz", flush=True)

    def score(result):
        a = result.analysis
        return {"load": a.failure_load, "load-cost": a.load_cost,
                "lower-load": a.lower_failure_load}[args.objective]

    best = max(results, key=score)
    best.truss = replace(best.truss, name="Optimized candidate")
    best.analysis = solve(best.truss, load_oz=args.load)
    _result(best.analysis, out / "best", args)
    write_json(out / "history.json", best.history)
    write_json(out / "metadata.json", best.metadata)
    write_csv(out / "runs.csv", [
        {"seed": seed, "nominal_failure_load_oz": result.analysis.failure_load,
         "lower_envelope_oz": result.analysis.lower_failure_load,
         "cost": result.analysis.cost, "load_cost": result.analysis.load_cost,
         "objective": args.objective, "objective_value": score(result)}
        for seed, result in zip(args.seeds, results)
    ])
    if not args.no_figures:
        plot_optimization(best.history, truss, out / "progress")
    return best


def _common(parser):
    parser.add_argument("--out", type=Path, required=True, help="Directory for generated results")
    parser.add_argument("--load", type=float, default=30.0, help="Applied downward load in oz-force (default: 30)")
    parser.add_argument("--kerf", type=float, default=0.0, help="Material lost per cut in inches (default: 0)")
    parser.add_argument("--team", default="Not specified", help="Team names included in analysis output")
    parser.add_argument("--section", default="Not specified", help="Course section included in analysis output")
    parser.add_argument("--no-figures", action="store_true", help="Export numeric tables only")


def _search_options(parser):
    parser.add_argument("--generations", type=int, default=150, help="Maximum differential-evolution generations per run")
    parser.add_argument("--population", type=int, default=12, help="Population size multiplier per free variable")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Independent reproducible search seeds")
    parser.add_argument("--objective", choices=["load", "load-cost", "lower-load"], default="load",
                        help="Maximize nominal capacity, load/cost, or lower fit-envelope capacity")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate, analyze, optimize, and illustrate planar trusses.")
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate", help="Write editable geometry JSON")
    generate.add_argument("--kind", choices=["modified-warren", "equilateral"], default="modified-warren")
    generate.add_argument("--span", type=float, default=30.0)
    generate.add_argument("--panels", type=int, default=3)
    generate.add_argument("--load-height", type=float, default=1.25)
    generate.add_argument("--out", required=True, type=Path, help="Output JSON file")

    analyze = sub.add_parser("analyze", help="Solve and export all report tables and figures")
    analyze.add_argument("design", type=Path)
    _common(analyze)
    optimize = sub.add_parser("optimize", help="Optimize geometry for the supplied connectivity")
    optimize.add_argument("design", type=Path)
    _common(optimize)
    _search_options(optimize)
    compare = sub.add_parser("compare", help="Compare existing designs without rerunning optimization")
    compare.add_argument("designs", nargs="+", type=Path)
    _common(compare)
    history = sub.add_parser("animate", help="Regenerate progress plot and animation from saved generations")
    history.add_argument("design", type=Path, help="Original geometry JSON (for member connectivity)")
    history.add_argument("history", type=Path)
    history.add_argument("--out", type=Path, required=True)
    demo = sub.add_parser("demo", help="Run baseline, optimization, comparison, and report exports")
    _common(demo)
    _search_options(demo)
    args = parser.parse_args(argv)

    try:
        if hasattr(args, "load") and (not math.isfinite(args.load) or args.load <= 0):
            raise ValueError("--load must be finite and positive")
        if hasattr(args, "kerf") and args.kerf < 0:
            raise ValueError("--kerf must be nonnegative")
        if hasattr(args, "seeds") and len(set(args.seeds)) != len(args.seeds):
            raise ValueError("Use distinct --seeds so saved runs are not overwritten")
        if hasattr(args, "seeds") and any(seed < 0 for seed in args.seeds):
            raise ValueError("--seeds must be nonnegative integers")
        if args.command == "generate":
            truss = warren(args.span, args.panels, args.kind == "modified-warren", args.load_height)
            write_truss(args.out, truss)
            print(f"Wrote {args.out.resolve()}")
            if args.kind == "equilateral":
                print("Reference only: exact equilateral geometry violates the load-joint height rule.")
        elif args.command == "analyze":
            summary = _result(solve(read_truss(args.design), args.load), args.out, args)
            return 0 if summary["passes_all_checks"] else 2
        elif args.command == "optimize":
            _optimize(read_truss(args.design), args.out, args)
        elif args.command == "compare":
            analyses = [solve(read_truss(path), args.load) for path in args.designs]
            summaries = [_result(a, args.out / f"design-{i+1}", args) for i, a in enumerate(analyses)]
            write_json(args.out / "comparison.json", summaries)
            if not args.no_figures:
                plot_comparison(analyses, args.out)
            return 0 if all(s["passes_all_checks"] for s in summaries) else 2
        elif args.command == "animate":
            plot_optimization(json.loads(args.history.read_text()), read_truss(args.design), args.out)
            print(f"Wrote progress figures to {args.out.resolve()}")
        elif args.command == "demo":
            baseline = warren(raised_load=True, load_height=1.25)
            initial = solve(baseline, args.load)
            _result(initial, args.out / "baseline", args)
            best = _optimize(baseline, args.out / "optimization", args)
            if not args.no_figures:
                plot_comparison([initial, best.analysis], args.out / "comparison")
            write_json(args.out / "run.json", {
                "command": "demo", "seeds": args.seeds, "generations": args.generations,
                "population": args.population, "objective": args.objective,
                "environment": environment(),
                "baseline": "baseline/geometry.json",
                "best": "optimization/best/geometry.json",
                "scope": "Fixed connectivity; best feasible sampled geometry, not a global-optimum proof.",
            })
        return 0
    except (ValueError, KeyError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
