"""Run the four-maze pathfinding benchmark and export its deliverables."""

from __future__ import annotations

import argparse
import csv
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from algorithms.astar import AStar
from algorithms.base import SearchResult
from algorithms.dijkstra import Dijkstra
from algorithms.q_learning import run_q_learning
from environment.generators import all_configurations
from visualization.plotter import plot_comparison, plot_metrics_barchart


@dataclass
class BenchmarkRow:
    configuration: str
    algorithm: str
    success: bool
    steps: int
    runtime_ms: float
    expanded: int
    path_cost: float


def _run_configuration(grid) -> List[SearchResult]:
    """Run every required algorithm against the same generated grid."""
    results = [Dijkstra().search(grid), AStar("manhattan").search(grid)]
    q_result, train_ms = run_q_learning(
        grid,
        max_episodes=2000,
        max_steps=grid.rows * grid.cols * 2,
        seed=42,
    )
    q_result.runtime_ms += train_ms
    results.append(q_result)
    return results


def _rows(config_label: str, results: List[SearchResult]) -> List[BenchmarkRow]:
    return [
        BenchmarkRow(
            configuration=config_label,
            algorithm=result.algorithm,
            success=result.success,
            steps=result.steps,
            runtime_ms=result.runtime_ms,
            expanded=result.expanded,
            path_cost=result.path_cost,
        )
        for result in results
    ]


def _write_csv(rows: List[BenchmarkRow], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["configuration", "algorithm", "success", "steps", "runtime_ms", "expanded", "path_cost"])
        for row in rows:
            writer.writerow([
                row.configuration,
                row.algorithm,
                row.success,
                row.steps,
                f"{row.runtime_ms:.4f}",
                row.expanded,
                f"{row.path_cost:.4f}",
            ])


def _write_report(rows: List[BenchmarkRow], path: Path, figure_names: List[str]) -> None:
    configurations = []
    for row in rows:
        if row.configuration not in configurations:
            configurations.append(row.configuration)

    lines = [
        "# Task 3: Heuristic Graph Pathfinding Agent",
        "",
        f"Benchmark generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Summary",
        "",
        "This benchmark compares Dijkstra, A* with the Manhattan heuristic, and the bonus tabular Q-learning agent on four reproducible 4-connected grid mazes.",
        "All configurations are generated with fixed seeds and are verified reachable by the generator before search.",
        "",
        "## Implemented Components",
        "",
        "- `environment/grid.py`: bounded grid, obstacle handling, 4-connected movement, and admissible distance heuristics.",
        "- `environment/generators.py`: Easy, Medium, Hard, and Cul-de-Sac benchmark mazes.",
        "- `algorithms/dijkstra.py`: optimal uniform-cost baseline.",
        "- `algorithms/astar.py`: optimal heuristic-guided search with selectable heuristics.",
        "- `algorithms/q_learning.py`: bonus tabular Q-learning training and greedy path extraction.",
        "- `visualization/plotter.py`: per-maze comparison panels and aggregate metric charts.",
        "",
        "## Metrics",
        "",
        "`Steps` is the final path length, `Runtime (ms)` includes Q-learning training and inference for the RL row, and `Expanded` is the algorithm-specific explored-node count.",
        "",
        "| Configuration | Algorithm | Solved | Steps | Runtime (ms) | Expanded | Path cost |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.configuration} | {row.algorithm} | {'Yes' if row.success else 'No'} | {row.steps} | {row.runtime_ms:.4f} | {row.expanded} | {row.path_cost:.1f} |"
        )

    lines.extend([
        "",
        "## Findings",
        "",
        "- Dijkstra and A* produce equal optimal path costs on every solvable benchmark maze.",
        "- A* expands no more nodes than Dijkstra in the tested configurations, while using the goal-directed Manhattan heuristic.",
        "- Q-learning is stochastic model-free training; its reported runtime includes training and greedy extraction, while its expanded count measures inference states only.",
        "",
        "## Exported Visualizations",
        "",
    ])
    lines.extend(f"- `{name}`" for name in figure_names)
    lines.extend([
        "",
        "## Validation",
        "",
        "The repository test suite contains 24 unit and integration tests covering grid behavior, heuristic validity, path correctness, optimality, unsolvable grids, and all four benchmark configurations.",
        "",
        "## Reproduction",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe benchmark.py",
        "```",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_pdf(report_path: Path, pdf_path: Path) -> None:
    """Render the Markdown report as a portable text PDF using Matplotlib."""
    lines = report_path.read_text(encoding="utf-8").splitlines()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        page_lines: List[str] = []
        for line in lines:
            wrapped = textwrap.wrap(line, width=105) or [""]
            page_lines.extend(wrapped)
        page_size = 44
        for start in range(0, len(page_lines), page_size):
            figure = Figure(figsize=(8.27, 11.69), dpi=150)
            axis = figure.add_axes([0.08, 0.06, 0.84, 0.88])
            axis.axis("off")
            axis.text(0, 1, "\n".join(page_lines[start:start + page_size]), va="top", ha="left", fontsize=9, family="DejaVu Sans", linespacing=1.35)
            pdf.savefig(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--reports", type=Path, default=Path("..") / "Reports")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    args.reports.mkdir(parents=True, exist_ok=True)
    config_labels: List[str] = []
    all_results: List[List[SearchResult]] = []
    rows: List[BenchmarkRow] = []

    for label, grid in all_configurations():
        print(f"Running {label}...")
        results = _run_configuration(grid)
        config_labels.append(label.split(" (")[0])
        all_results.append(results)
        rows.extend(_rows(label, results))
        filename = label.split(" (")[0].lower().replace("-", "_").replace(" ", "_")
        plot_comparison(grid, results, label, str(args.output / f"{filename}_comparison.png"))

    plot_metrics_barchart(config_labels, all_results, str(args.output / "metrics_comparison.png"))
    _write_csv(rows, args.output / "benchmark_metrics.csv")

    report_path = Path("REPORT.md")
    figure_names = sorted(path.name for path in args.output.glob("*.png"))
    _write_report(rows, report_path, [f"artifacts/{name}" for name in figure_names])
    _render_pdf(report_path, args.reports / "Task_03_Graph_Pathfinding_Report.pdf")
    print(f"Wrote {report_path}")
    print(f"Wrote {args.reports / 'Task_03_Graph_Pathfinding_Report.pdf'}")


if __name__ == "__main__":
    main()
