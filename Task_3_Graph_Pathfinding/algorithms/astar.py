"""
A* Search Algorithm — implemented from scratch.

A* is an INFORMED heuristic pathfinding algorithm that combines the exact
cumulative cost g(n) with an admissible heuristic estimate h(n) of the
remaining distance to the goal, evaluating each node by:

    f(n) = g(n) + h(n)

where:
    g(n) = exact cost from the start node to node n (same as Dijkstra)
    h(n) = admissible heuristic estimate from n to goal
           (never overestimates the true remaining cost)

Because h(n) guides the search TOWARD the goal, A* expands drastically
fewer nodes than Dijkstra on most practical grids while still guaranteeing
an optimal path (when h is admissible and consistent).

Supported Heuristics (selectable at construction time):
    "manhattan"  — L₁ norm: |Δrow| + |Δcol|  (optimal for 4-connected grids)
    "euclidean"  — L₂ norm: sqrt(Δrow² + Δcol²)
    "chebyshev"  — L∞ norm: max(|Δrow|, |Δcol|)

Algorithm Overview:
  1. Initialise g(start) = 0, f(start) = h(start), push (f, start) to heap.
  2. While open-set is not empty:
     a. Pop node n with the smallest f(n).
     b. If n == goal → success; reconstruct and return the path.
     c. For each passable neighbour m of n:
        - new_g = g(n) + cost(n, m)
        - If new_g < g(m): update g(m), f(m) = new_g + h(m),
          record predecessor, push (f(m), m) to heap.
  3. If open-set exhausted → no path exists.

Time  Complexity: O(E log V)  in the best case with perfect heuristic
Space Complexity: O(V)
"""

from __future__ import annotations
import heapq
import time
from typing import Callable, Dict, Set

from environment.grid import Grid, Cell
from .base import PathfindingAlgorithm, SearchResult


# Heuristic function type alias
HeuristicFn = Callable[[Cell, Cell], float]


class AStar(PathfindingAlgorithm):
    """
    A* Search with a configurable admissible heuristic.

    Parameters
    ----------
    heuristic : str
        One of 'manhattan', 'euclidean', or 'chebyshev'.
        Defaults to 'manhattan' (optimal for 4-connected uniform-cost grids).
    """

    HEURISTICS: Dict[str, HeuristicFn] = {
        "manhattan": Grid.manhattan,
        "euclidean": Grid.euclidean,
        "chebyshev": Grid.chebyshev,
    }

    def __init__(self, heuristic: str = "manhattan"):
        """
        Args:
            heuristic: Name of the admissible heuristic function to use.
        """
        if heuristic not in self.HEURISTICS:
            raise ValueError(
                f"Unknown heuristic '{heuristic}'. "
                f"Choose from: {list(self.HEURISTICS)}"
            )
        self.heuristic_name = heuristic
        self._h: HeuristicFn = self.HEURISTICS[heuristic]
        super().__init__(name=f"A* ({heuristic.capitalize()})")

    def search(self, grid: Grid) -> SearchResult:
        """
        Execute A* search on *grid*.

        Args:
            grid: Configured Grid with grid.start and grid.goal set.

        Returns:
            SearchResult with the optimal path and performance metrics.
        """
        assert grid.start is not None, "Grid must have a start cell."
        assert grid.goal  is not None, "Grid must have a goal cell."

        start = grid.start
        goal  = grid.goal
        h     = self._h          # heuristic function shorthand
        t0    = time.perf_counter()

        # --- Priority queue entries: (f_cost, tie-breaking counter, cell) ---
        # The counter breaks ties when f values are equal, preventing
        # Python from comparing Cell tuples which could cause errors.
        counter = 0
        open_set: list = [(h(start, goal), counter, start)]

        # g[n] = best known cumulative cost from start to n
        g: Dict[Cell, float] = {start: 0.0}

        # came_from[n] = predecessor cell on the best path to n
        came_from: Dict[Cell, Cell] = {}

        # Closed set of fully expanded nodes
        closed:   Set[Cell] = set()
        explored: Set[Cell] = {start}

        expanded = 0

        while open_set:
            # Step 2a: pop node with smallest f(n) = g(n) + h(n)
            f_current, _, current = heapq.heappop(open_set)
            expanded += 1

            # Lazy deletion: skip stale heap entries
            if current in closed:
                continue
            closed.add(current)

            # Step 2b: goal check
            if current == goal:
                return self._make_result(
                    success=True,
                    came_from=came_from,
                    goal=goal,
                    g_cost=g[current],
                    expanded=expanded,
                    explored=explored,
                    start_time=t0,
                )

            # Step 2c: expand neighbours
            for neighbour in grid.neighbours(current):
                if neighbour in closed:
                    continue

                # Compute tentative g-cost through current node
                new_g = g[current] + grid.move_cost(current, neighbour)

                if new_g < g.get(neighbour, float("inf")):
                    # Better path to neighbour found → update tables
                    g[neighbour]         = new_g
                    came_from[neighbour] = current
                    explored.add(neighbour)

                    # f(n) = g(n) + h(n, goal)
                    f_neighbour = new_g + h(neighbour, goal)
                    counter += 1
                    heapq.heappush(open_set, (f_neighbour, counter, neighbour))

        # Step 3: no path found
        return self._make_result(
            success=False,
            came_from=came_from,
            goal=goal,
            g_cost=float("inf"),
            expanded=expanded,
            explored=explored,
            start_time=t0,
        )
