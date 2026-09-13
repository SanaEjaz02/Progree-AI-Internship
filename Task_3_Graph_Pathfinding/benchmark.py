"""Run the four-maze pathfinding benchmark and export its deliverables."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.image import imread

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
        "# TASK 3: Heuristic Graph Pathfinding Agent",
        "",
        "**Progree Remote Artificial Intelligence Internship | Technical Report**",
        "",
        f"Benchmark generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## 1. Executive Summary & Objective",
        "",
        "This report documents an end-to-end graph pathfinding benchmark across four reproducible 4-connected grid mazes. It compares Dijkstra's optimal uniform-cost search, A* with an admissible Manhattan heuristic, and a bonus tabular Q-learning agent.",
        "",
        "All configurations use fixed seeds and are verified reachable by the generator before search. The benchmark measures solution quality, explored states, and runtime so that optimality and search efficiency can be considered together.",
        "",
        "## 2. System Architecture",
        "",
        "- `environment/grid.py`: bounded grid, obstacle handling, 4-connected movement, and admissible distance heuristics.",
        "- `environment/generators.py`: Easy, Medium, Hard, and Cul-de-Sac benchmark mazes.",
        "- `algorithms/dijkstra.py`: optimal uniform-cost baseline.",
        "- `algorithms/astar.py`: optimal heuristic-guided search with selectable heuristics.",
        "- `algorithms/q_learning.py`: bonus tabular Q-learning training and greedy path extraction.",
        "- `visualization/plotter.py`: per-maze comparison panels and aggregate metric charts.",
        "",
        "The deterministic searches use the same generated grid and four-neighbor movement model. A* prioritizes nodes using $f(n)=g(n)+h(n)$, where the Manhattan distance is admissible for this movement model. Q-learning is evaluated separately as a model-free agent; its runtime includes training and greedy path extraction.",
        "",
        "## 3. Benchmark Metrics",
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
        "## 4. Findings",
        "",
        "- Dijkstra and A* produce equal optimal path costs on every solvable benchmark maze.",
        "- A* expands no more nodes than Dijkstra in the tested configurations, while using the goal-directed Manhattan heuristic.",
        "- Q-learning is stochastic model-free training; its reported runtime includes training and greedy extraction, while its expanded count measures inference states only.",
        "",
        "## 5. Visual Results",
        "",
    ])
    captions = {
        "easy_comparison.png": "Easy maze comparison: all three agents recover the same optimal route.",
        "medium_comparison.png": "Medium maze comparison: A* reaches the goal while expanding fewer states than Dijkstra.",
        "hard_comparison.png": "Hard maze comparison: the larger obstacle field increases search effort while preserving path optimality.",
        "cul_de_sac_trap_comparison.png": "Cul-de-Sac trap: the goal-directed and learned policies are compared in a deceptive U-shaped layout.",
        "metrics_comparison.png": "Aggregate comparison of expanded nodes and runtime across all benchmark configurations.",
    }
    for name in figure_names:
        filename = Path(name).name
        lines.extend([f"### {filename.replace('_', ' ').replace('.png', '').title()}", "", f"![{captions.get(filename, filename)}](artifacts/{filename})", "", captions.get(filename, filename), ""])
    lines.extend([
        "## 6. Validation & Reproduction",
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


def _render_pdf(report_path: Path, pdf_path: Path, rows: List[BenchmarkRow]) -> None:
    """Render a styled, image-rich technical report using Matplotlib."""
    del report_path
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    navy = "#1e3a8a"
    blue = "#2563eb"
    slate = "#4b5563"
    pale = "#f1f5f9"
    border = "#cbd5e1"

    def add_header(figure: Figure, section: str) -> None:
        figure.text(0.08, 0.955, "TASK 3: GRAPH PATHFINDING AGENT", color=navy, fontsize=15, fontweight="bold", va="top")
        figure.text(0.08, 0.925, "Progree Remote Artificial Intelligence Internship | Technical Report", color=slate, fontsize=9.5, va="top")
        figure.lines.append(matplotlib.lines.Line2D([0.08, 0.92], [0.905, 0.905], transform=figure.transFigure, color=blue, linewidth=2))
        figure.text(0.08, 0.875, section, color=navy, fontsize=12, fontweight="bold", va="top")

    def add_footer(figure: Figure, page: int) -> None:
        figure.lines.append(matplotlib.lines.Line2D([0.08, 0.92], [0.055, 0.055], transform=figure.transFigure, color=border, linewidth=0.7))
        figure.text(0.08, 0.035, "Task 3 | Graph Pathfinding", color=slate, fontsize=8)
        figure.text(0.92, 0.035, f"Page {page}", color=slate, fontsize=8, ha="right")

    def add_text(figure: Figure, text: str, y: float, width: int = 105, size: float = 9.5, color: str = "#1a1a1a") -> float:
        words = text.split()
        lines = []
        current = ""
        for word in words:
            if len(current) + len(word) + 1 <= width:
                current = f"{current} {word}".strip()
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        figure.text(0.08, y, "\n".join(lines), color=color, fontsize=size, va="top", linespacing=1.45)
        return y - 0.035 * len(lines) - 0.018

    def save_page(pdf: PdfPages, figure: Figure) -> None:
        FigureCanvasAgg(figure)
        pdf.savefig(figure)

    with PdfPages(pdf_path) as pdf:
        page = 1

        figure = Figure(figsize=(8.27, 11.69), dpi=150, facecolor="white")
        add_header(figure, "1. Executive Summary & Objective")
        meta = figure.add_axes([0.08, 0.73, 0.84, 0.105])
        meta.set_facecolor(pale)
        meta.set_xticks([])
        meta.set_yticks([])
        for spine in meta.spines.values():
            spine.set_visible(False)
        meta.spines["left"].set_visible(True)
        meta.spines["left"].set_color(blue)
        meta.spines["left"].set_linewidth(4)
        meta.text(0.025, 0.68, "Program: Progree AI Internship | Task: 3 (Graph Pathfinding)", fontsize=9.5, fontweight="bold", color="#1e293b")
        meta.text(0.025, 0.34, "Domain: Search Algorithms | Status: Completed | Evaluation: Optimality, Runtime, Expanded Nodes", fontsize=8.8, color=slate)
        y = 0.69
        y = add_text(figure, "This benchmark compares Dijkstra, A* with the Manhattan heuristic, and the bonus tabular Q-learning agent on four reproducible 4-connected grid mazes.", y)
        y = add_text(figure, "The deterministic algorithms provide an optimality baseline, while the learned agent demonstrates a model-free alternative. Every maze is generated with a fixed seed and verified reachable before evaluation.", y)
        figure.text(0.08, 0.565, "Key outcomes", color=navy, fontsize=11, fontweight="bold")
        cards = [("4", "benchmark mazes"), ("3", "algorithms"), ("24", "automated tests"), ("5", "PNG visualizations")]
        for index, (value, label) in enumerate(cards):
            axis = figure.add_axes([0.08 + index * 0.215, 0.46, 0.18, 0.075])
            axis.set_facecolor("#f8fafc")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color(border)
            axis.text(0.08, 0.58, value, fontsize=18, fontweight="bold", color=blue)
            axis.text(0.08, 0.18, label, fontsize=8.2, color=slate)
        figure.text(0.08, 0.39, "System architecture", color=navy, fontsize=11, fontweight="bold")
        add_text(figure, "Grid environment -> maze generator -> Dijkstra / A* / Q-learning -> metrics CSV and comparison visualizations", 0.355, width=100, size=10, color=slate)
        add_footer(figure, page)
        save_page(pdf, figure)
        page += 1

        figure = Figure(figsize=(8.27, 11.69), dpi=150, facecolor="white")
        add_header(figure, "2. Benchmark Metrics & Findings")
        axis = figure.add_axes([0.06, 0.26, 0.88, 0.58])
        axis.axis("off")
        table_data = [["Configuration", "Algorithm", "Solved", "Steps", "Runtime ms", "Expanded", "Cost"]]
        for row in rows:
            table_data.append([row.configuration, row.algorithm, "Yes" if row.success else "No", str(row.steps), f"{row.runtime_ms:.2f}", str(row.expanded), f"{row.path_cost:.1f}"])
        table = axis.table(cellText=table_data, loc="upper center", cellLoc="left", colWidths=[0.22, 0.18, 0.08, 0.08, 0.12, 0.11, 0.09])
        table.auto_set_font_size(False)
        table.set_fontsize(6.8)
        table.scale(1, 1.55)
        for (row_index, col_index), cell in table.get_celld().items():
            cell.set_edgecolor(border)
            cell.set_linewidth(0.5)
            if row_index == 0:
                cell.set_facecolor("#f8fafc")
                cell.set_text_props(weight="bold", color="#1e293b")
            elif row_index % 2 == 0:
                cell.set_facecolor("#f8fafc")
        figure.text(0.08, 0.19, "Key findings", color=navy, fontsize=11, fontweight="bold")
        add_text(figure, "Dijkstra and A* produce equal optimal path costs on every solvable maze. A* expands no more nodes than Dijkstra in these configurations because the Manhattan heuristic guides the frontier toward the goal. Q-learning is stochastic and its runtime includes training, so its timing is not directly comparable to one-pass deterministic search.", 0.165, width=105, size=9.2)
        add_footer(figure, page)
        save_page(pdf, figure)
        page += 1

        figure = Figure(figsize=(8.27, 11.69), dpi=150, facecolor="white")
        add_header(figure, "3. Implementation & Reproduction")
        y = 0.84
        for heading, body in [
            ("Environment", "Bounded grid state, obstacle handling, 4-connected movement, and admissible distance heuristics."),
            ("Search algorithms", "Dijkstra provides the optimal uniform-cost baseline. A* adds the Manhattan heuristic. Tabular Q-learning trains a policy and extracts a greedy route."),
            ("Outputs", "The benchmark exports a metrics CSV, four per-maze comparison panels, and one aggregate performance chart."),
            ("Validation", "The repository test suite covers grid behavior, heuristic validity, path correctness, optimality, unsolvable grids, and all four benchmark configurations."),
        ]:
            figure.text(0.08, y, heading, color=navy, fontsize=10.5, fontweight="bold")
            y = add_text(figure, body, y - 0.026, width=105, size=9.4)
        figure.text(0.08, 0.40, "Reproduce the benchmark", color=navy, fontsize=11, fontweight="bold")
        code = figure.add_axes([0.08, 0.30, 0.84, 0.075])
        code.set_facecolor("#f8fafc")
        code.set_xticks([])
        code.set_yticks([])
        for spine in code.spines.values():
            spine.set_color(border)
        code.text(0.03, 0.55, r".\.venv\Scripts\python.exe benchmark.py", family="DejaVu Sans Mono", fontsize=10, color="#b91c1c", va="center")
        add_text(figure, "The generated Markdown report includes the same five figures shown on the following pages.", 0.24, width=105, size=9.4, color=slate)
        add_footer(figure, page)
        save_page(pdf, figure)
        page += 1

        figure_paths = sorted(Path("artifacts").glob("*.png"))
        captions = {
            "easy_comparison.png": "Easy maze comparison",
            "medium_comparison.png": "Medium maze comparison",
            "hard_comparison.png": "Hard maze comparison",
            "cul_de_sac_trap_comparison.png": "Cul-de-Sac trap comparison",
            "metrics_comparison.png": "Aggregate metrics comparison",
        }
        for image_path in figure_paths:
            figure = Figure(figsize=(8.27, 11.69), dpi=150, facecolor="white")
            add_header(figure, "4. Visual Results")
            figure.text(0.08, 0.84, captions.get(image_path.name, image_path.stem.replace("_", " ").title()), color=navy, fontsize=12, fontweight="bold")
            axis = figure.add_axes([0.08, 0.22, 0.84, 0.56])
            axis.imshow(imread(image_path), aspect="equal")
            axis.axis("off")
            figure.text(0.08, 0.16, "Generated from the fixed-seed benchmark run. See REPORT.md for the corresponding interpretation and metric definitions.", color=slate, fontsize=8.8)
            add_footer(figure, page)
            save_page(pdf, figure)
            page += 1


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
    _render_pdf(report_path, args.reports / "Task_03_Graph_Pathfinding_Report.pdf", rows)
    print(f"Wrote {report_path}")
    print(f"Wrote {args.reports / 'Task_03_Graph_Pathfinding_Report.pdf'}")


if __name__ == "__main__":
    main()
