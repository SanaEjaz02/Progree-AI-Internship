"""
Visualization Module for Heuristic Pathfinding Agent.

Renders high-resolution grid maps with Matplotlib showing:
  - Obstacle walls (dark grey)
  - Start cell (bright green)
  - Goal cell  (bright red)
  - Explored / frontier cells (translucent blue heatmap)
  - Final optimal path (gold/yellow line overlay)

Exports:
  - Per-maze side-by-side comparison panels (Dijkstra vs A* vs Q-Learning)
  - Aggregate metric bar chart (runtime and expanded nodes)
"""

from __future__ import annotations
import os
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server/script use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from environment.grid import Grid, Cell
from algorithms.base import SearchResult


# ---------------------------------------------------------------------------
# Colour constants
# ---------------------------------------------------------------------------
COLOUR_FREE      = (0.96, 0.96, 0.96)   # very light grey
COLOUR_OBSTACLE  = (0.22, 0.22, 0.22)   # dark charcoal
COLOUR_EXPLORED  = (0.53, 0.81, 0.98)   # light sky blue
COLOUR_PATH      = (1.00, 0.84, 0.00)   # gold
COLOUR_START     = (0.20, 0.78, 0.35)   # emerald green
COLOUR_GOAL      = (0.92, 0.26, 0.21)   # crimson red


def _build_grid_image(grid: Grid, result: Optional[SearchResult]) -> np.ndarray:
    """
    Build a (rows × cols × 3) RGB image array representing the grid state.

    Cell colouring priority:
        1. Obstacle  → dark grey
        2. Explored  → sky blue
        3. Path      → gold
        4. Start     → green
        5. Goal      → red
        6. Free      → off-white
    """
    rows, cols = grid.rows, grid.cols
    img = np.ones((rows, cols, 3), dtype=float)

    # Fill free cells
    for r in range(rows):
        for c in range(cols):
            img[r, c] = COLOUR_FREE

    # Obstacles
    for r in range(rows):
        for c in range(cols):
            if grid.is_obstacle((r, c)):
                img[r, c] = COLOUR_OBSTACLE

    if result is not None:
        # Explored frontier
        if result.explored:
            for (r, c) in result.explored:
                if 0 <= r < rows and 0 <= c < cols:
                    if not grid.is_obstacle((r, c)):
                        img[r, c] = COLOUR_EXPLORED

        # Optimal path
        if result.path:
            for (r, c) in result.path:
                if 0 <= r < rows and 0 <= c < cols:
                    img[r, c] = COLOUR_PATH

    # Start and Goal always on top
    if grid.start:
        sr, sc = grid.start
        img[sr, sc] = COLOUR_START
    if grid.goal:
        gr, gc = grid.goal
        img[gr, gc] = COLOUR_GOAL

    return img


def plot_comparison(
    grid: Grid,
    results: List[SearchResult],
    title: str,
    save_path: str
) -> None:
    """
    Render a side-by-side comparison panel showing the grid state after each
    algorithm, and save to *save_path* as a PNG.

    Args:
        grid      : The grid environment shared by all algorithms.
        results   : List of SearchResult objects (one per algorithm).
        title     : Overall panel title (e.g. maze difficulty name).
        save_path : Absolute or relative file path for the exported PNG.
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.5))
    if n == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    # Shared legend patches
    legend_patches = [
        mpatches.Patch(color=COLOUR_OBSTACLE, label="Wall"),
        mpatches.Patch(color=COLOUR_START,    label="Start"),
        mpatches.Patch(color=COLOUR_GOAL,     label="Goal"),
        mpatches.Patch(color=COLOUR_EXPLORED, label="Explored"),
        mpatches.Patch(color=COLOUR_PATH,     label="Optimal Path"),
    ]

    for ax, result in zip(axes, results):
        img = _build_grid_image(grid, result)
        ax.imshow(img, interpolation="nearest", aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])

        # Sub-title with key metrics
        if result.success:
            sub = (
                f"{result.algorithm}\n"
                f"Steps: {result.steps}  |  Expanded: {result.expanded}\n"
                f"Time: {result.runtime_ms:.2f} ms"
            )
        else:
            sub = f"{result.algorithm}\nNO PATH FOUND"

        ax.set_title(sub, fontsize=9, pad=4)

    # Add shared legend below the panels
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=5,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.05),
        framealpha=0.9
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {save_path}")


def plot_metrics_barchart(
    config_labels: List[str],
    all_results: List[List[SearchResult]],
    save_path: str
) -> None:
    """
    Render grouped bar charts comparing Expanded Nodes and Runtime across
    all maze configurations and algorithms.

    Args:
        config_labels : List of maze configuration names (x-axis groups).
        all_results   : Outer list per configuration, inner list per algorithm.
        save_path     : Output PNG path.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Algorithm Performance Comparison Across Maze Configurations",
        fontsize=13, fontweight="bold"
    )

    # Gather unique algorithm names
    alg_names = [r.algorithm for r in all_results[0]]
    n_algs    = len(alg_names)
    n_configs = len(config_labels)

    x       = np.arange(n_configs)
    width   = 0.8 / n_algs
    colours = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    for i, alg_name in enumerate(alg_names):
        expanded_vals = [
            cfg_results[i].expanded for cfg_results in all_results
        ]
        runtime_vals = [
            cfg_results[i].runtime_ms for cfg_results in all_results
        ]
        offset = (i - n_algs / 2 + 0.5) * width
        colour = colours[i % len(colours)]

        ax1.bar(x + offset, expanded_vals, width, label=alg_name, color=colour, alpha=0.88)
        ax2.bar(x + offset, runtime_vals,  width, label=alg_name, color=colour, alpha=0.88)

    # Format Expanded Nodes chart
    ax1.set_title("Nodes Expanded", fontsize=11)
    ax1.set_ylabel("Number of Expanded Nodes")
    ax1.set_xticks(x)
    ax1.set_xticklabels(config_labels, rotation=15, ha="right", fontsize=8)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Format Runtime chart
    ax2.set_title("Runtime (milliseconds)", fontsize=11)
    ax2.set_ylabel("Time (ms)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(config_labels, rotation=15, ha="right", fontsize=8)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {save_path}")
