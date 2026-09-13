"""
Dijkstra's Shortest-Path Algorithm — implemented from scratch.

Dijkstra's algorithm is an INFORMED (over cost) but UNINFORMED (over goal
position) single-source shortest-path algorithm.  It expands nodes in
non-decreasing order of their cumulative cost g(n) from the start node,
guaranteeing the optimal path but potentially exploring a large portion of
the grid before reaching the goal.

Algorithm Overview:
  1. Initialise g(start) = 0, push (0, start) to the min-heap open-set.
  2. While open-set is not empty:
     a. Pop the node n with the smallest g(n).
     b. If n == goal → success; reconstruct and return the path.
     c. For each passable neighbour m of n:
        - new_g = g(n) + cost(n, m)
        - If new_g < g(m): update g(m), record predecessor, push to heap.
  3. If open-set exhausted → no path exists.

Time  Complexity: O((V + E) log V)  where V = cells, E = passable edges
Space Complexity: O(V)
"""

from __future__ import annotations
import heapq
import time
from typing import Dict, Set

from environment.grid import Grid, Cell
from .base import PathfindingAlgorithm, SearchResult


class Dijkstra(PathfindingAlgorithm):
    """
    Dijkstra's uniform-cost pathfinding algorithm.

    Expands nodes purely by accumulated cost g(n), making no assumptions
    about the direction of the goal.  Serves as the unguided optimal
    baseline against which A* is compared.
    """

    def __init__(self):
        super().__init__(name="Dijkstra")

    def search(self, grid: Grid) -> SearchResult:
        """
        Execute Dijkstra's algorithm on *grid*.

        Args:
            grid: Configured Grid with grid.start and grid.goal set.

        Returns:
            SearchResult with the shortest path and performance metrics.
        """
        assert grid.start is not None, "Grid must have a start cell."
        assert grid.goal  is not None, "Grid must have a goal cell."

        start = grid.start
        goal  = grid.goal
        t0    = time.perf_counter()

        # --- Priority queue entries: (g_cost, cell) ---
        # Python's heapq is a MIN-heap, so smallest g_cost is popped first.
        open_set: list = [(0.0, start)]

        # g[n] = best known cost from start to n
        g: Dict[Cell, float] = {start: 0.0}

        # came_from[n] = the node from which n was best reached
        came_from: Dict[Cell, Cell] = {}

        # Closed set: nodes already fully expanded (popped from open_set)
        closed:   Set[Cell] = set()
        explored: Set[Cell] = {start}

        expanded = 0  # count of nodes popped and processed

        while open_set:
            # Step 2a: pop node with smallest g value
            current_g, current = heapq.heappop(open_set)
            expanded += 1

            # Skip if we have already found a better path to this node
            # (lazy deletion — stale entries may remain in the heap)
            if current in closed:
                continue
            closed.add(current)

            # Step 2b: goal check
            if current == goal:
                return self._make_result(
                    success=True,
                    came_from=came_from,
                    goal=goal,
                    g_cost=current_g,
                    expanded=expanded,
                    explored=explored,
                    start_time=t0,
                )

            # Step 2c: expand neighbours
            for neighbour in grid.neighbours(current):
                if neighbour in closed:
                    continue

                # Compute tentative cost through current node
                new_g = current_g + grid.move_cost(current, neighbour)

                if new_g < g.get(neighbour, float("inf")):
                    # Found a better path to neighbour → update
                    g[neighbour]          = new_g
                    came_from[neighbour]  = current
                    explored.add(neighbour)
                    heapq.heappush(open_set, (new_g, neighbour))

        # Step 3: open_set exhausted — no path found
        return self._make_result(
            success=False,
            came_from=came_from,
            goal=goal,
            g_cost=float("inf"),
            expanded=expanded,
            explored=explored,
            start_time=t0,
        )
