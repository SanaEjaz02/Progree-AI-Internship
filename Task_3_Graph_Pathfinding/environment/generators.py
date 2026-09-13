"""
Maze / Grid Generators for Heuristic Pathfinding Benchmarks.

Produces 4 reproducible, verified-solvable grid configurations at increasing
difficulty levels to stress-test A* vs Dijkstra pathfinding performance:

  1. Easy      — 15×15,  ~10% obstacle density, sparse random scatter
  2. Medium    — 20×20,  ~25% obstacle density, structured blockade rows
  3. Hard      — 25×25,  ~35% obstacle density, winding labyrinth corridors
  4. Cul-de-Sac — 20×20, manually engineered U-trap forcing deep dead-end
                 exploration to highlight A* heuristic pruning advantage
"""

from __future__ import annotations
import random
from typing import List, Tuple, Optional

from .grid import Grid, Cell


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _place_start_goal(grid: Grid, start: Cell, goal: Cell) -> None:
    """
    Place start and goal on the grid, clearing any obstacles on those cells
    to guarantee they are always passable.
    """
    grid.start = start
    grid.goal  = goal
    grid.set_free(start)
    grid.set_free(goal)


def _is_reachable(grid: Grid) -> bool:
    """
    BFS connectivity check: return True if grid.goal is reachable from grid.start.

    Uses the grid's own neighbour generator so obstacles are respected.
    """
    from collections import deque
    if grid.start is None or grid.goal is None:
        return False

    visited = {grid.start}
    queue   = deque([grid.start])

    while queue:
        current = queue.popleft()
        if current == grid.goal:
            return True
        for nb in grid.neighbours(current):
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return False


def _random_obstacles(
    grid: Grid,
    density: float,
    rng: random.Random,
    protected: Optional[List[Cell]] = None
) -> None:
    """
    Randomly scatter obstacles across the grid at the requested density
    fraction, skipping any cells listed in *protected* (typically start/goal
    and any pre-built walls).
    """
    protected_set = set(protected) if protected else set()
    for r in range(grid.rows):
        for c in range(grid.cols):
            cell = (r, c)
            if cell not in protected_set and rng.random() < density:
                grid.set_obstacle(cell)


# ---------------------------------------------------------------------------
# Public generator functions
# ---------------------------------------------------------------------------

def make_easy_grid(seed: int = 42) -> Grid:
    """
    Easy configuration — 15×15 grid, ~10% random obstacle scatter.

    Agent starts at top-left corner and must reach bottom-right corner.
    Low obstacle density with a clear open path in most random seeds.
    """
    rows, cols = 15, 15
    grid  = Grid(rows, cols)
    start = (0, 0)
    goal  = (rows - 1, cols - 1)
    _place_start_goal(grid, start, goal)

    rng = random.Random(seed)
    # Scatter random obstacles at 10% density, avoiding start/goal
    _random_obstacles(grid, density=0.10, rng=rng, protected=[start, goal])

    # Guarantee solvability — keep regenerating if BFS fails
    attempts = 0
    while not _is_reachable(grid) and attempts < 20:
        grid = Grid(rows, cols)
        _place_start_goal(grid, start, goal)
        rng = random.Random(seed + attempts)
        _random_obstacles(grid, density=0.10, rng=rng, protected=[start, goal])
        attempts += 1

    return grid


def make_medium_grid(seed: int = 42) -> Grid:
    """
    Medium configuration — 20×20 grid, ~25% obstacle density with two
    horizontal blockade rows forcing the agent through narrow choke points.

    Tests whether A*'s heuristic guidance effectively navigates bottlenecks.
    """
    rows, cols = 20, 20
    grid  = Grid(rows, cols)
    start = (0, 0)
    goal  = (rows - 1, cols - 1)
    _place_start_goal(grid, start, goal)

    rng = random.Random(seed)

    # --- Structured blockade walls ---
    # Horizontal wall at row 6 with a single gap at column 10
    gap1 = 10
    for c in range(cols):
        if c != gap1:
            grid.set_obstacle((6, c))

    # Horizontal wall at row 13 with a single gap at column 4
    gap2 = 4
    for c in range(cols):
        if c != gap2:
            grid.set_obstacle((13, c))

    # Fill remaining free cells with ~20% random scatter
    protected = [start, goal]
    for r in range(rows):
        protected.append((6, gap1))
        protected.append((13, gap2))
    _random_obstacles(grid, density=0.20, rng=rng, protected=protected)

    attempts = 0
    while not _is_reachable(grid) and attempts < 20:
        grid = Grid(rows, cols)
        _place_start_goal(grid, start, goal)
        for c in range(cols):
            if c != gap1:
                grid.set_obstacle((6, c))
        for c in range(cols):
            if c != gap2:
                grid.set_obstacle((13, c))
        rng = random.Random(seed + attempts)
        _random_obstacles(grid, density=0.12, rng=rng,
                          protected=[(0, 0), goal, (6, gap1), (13, gap2)])
        attempts += 1

    return grid


def make_hard_grid(seed: int = 42) -> Grid:
    """
    Hard configuration — 25×25 grid, ~35% obstacle density with winding
    labyrinth-style corridor walls.

    Significantly stresses the search space; illustrates the node-expansion
    saving of A* over uninformed Dijkstra.
    """
    rows, cols = 25, 25
    grid  = Grid(rows, cols)
    start = (0, 0)
    goal  = (rows - 1, cols - 1)
    _place_start_goal(grid, start, goal)

    rng = random.Random(seed)

    # --- Staggered serpentine walls ---
    # Each pair of rows forms a winding corridor section
    wall_rows = [4, 9, 14, 19]
    gaps      = [cols - 3, 2, cols - 3, 2]   # alternating gap sides

    for wall_r, gap_c in zip(wall_rows, gaps):
        for c in range(cols):
            if c != gap_c:
                grid.set_obstacle((wall_r, c))

    # Random scatter at 20% on the remaining free cells
    protected_cells = [start, goal]
    for wall_r, gap_c in zip(wall_rows, gaps):
        protected_cells.append((wall_r, gap_c))

    _random_obstacles(grid, density=0.20, rng=rng, protected=protected_cells)

    attempts = 0
    while not _is_reachable(grid) and attempts < 30:
        grid = Grid(rows, cols)
        _place_start_goal(grid, start, goal)
        for wall_r, gap_c in zip(wall_rows, gaps):
            for c in range(cols):
                if c != gap_c:
                    grid.set_obstacle((wall_r, c))
        rng = random.Random(seed + attempts + 100)
        _random_obstacles(grid, density=0.12, rng=rng,
                          protected=protected_cells)
        attempts += 1

    return grid


def make_cul_de_sac_grid() -> Grid:
    """
    Deceptive Cul-de-Sac (U-Trap) configuration — 20×20 grid.

    A deep U-shaped dead-end is positioned directly along the shortest
    straight-line path from start (0,0) to goal (19,19).  Greedy/uninformed
    searches plunge deep into the trap; A*'s admissible heuristic detects the
    dead-end early and backtracks efficiently.

    This configuration starkly demonstrates the performance gap between
    heuristic-guided (A*) and uninformed (Dijkstra) search.
    """
    rows, cols = 20, 20
    grid  = Grid(rows, cols)
    start = (0, 0)
    goal  = (rows - 1, cols - 1)
    _place_start_goal(grid, start, goal)

    # --- U-shaped dead-end trap in the centre-left region ---
    # Top wall of the U
    for c in range(4, 13):
        grid.set_obstacle((4, c))
    # Bottom wall of the U
    for c in range(4, 13):
        grid.set_obstacle((10, c))
    # Left wall of the U (closing the bottom)
    for r in range(4, 11):
        grid.set_obstacle((r, 4))
    # The mouth of the U faces right — so the path into the trap is
    # (r, 12) for r in [5..9].  The exit corridor is only accessible
    # by going around the top.

    # Add a small random scatter outside the trap for realism
    rng = random.Random(7)
    for r in range(rows):
        for c in range(cols):
            cell = (r, c)
            if (not grid.is_obstacle(cell)
                    and cell != start
                    and cell != goal
                    and rng.random() < 0.06):
                grid.set_obstacle(cell)

    # Guarantee reachability
    if not _is_reachable(grid):
        # Fallback: clear a safe column along the right edge
        for r in range(rows):
            grid.set_free((r, cols - 1))

    return grid


# ---------------------------------------------------------------------------
# Convenience: all configurations as a named list
# ---------------------------------------------------------------------------

def all_configurations() -> List[Tuple[str, Grid]]:
    """
    Return a list of (label, grid) tuples for all benchmark configurations.
    """
    return [
        ("Easy (15×15, 10% density)",         make_easy_grid()),
        ("Medium (20×20, 25% density)",        make_medium_grid()),
        ("Hard (25×25, 35% density)",          make_hard_grid()),
        ("Cul-de-Sac Trap (20×20, U-shaped)", make_cul_de_sac_grid()),
    ]
