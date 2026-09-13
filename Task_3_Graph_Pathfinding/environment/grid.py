"""
Grid Environment for Heuristic Pathfinding Agent.

Defines the 2D discrete grid world, obstacle queries, neighbour generation,
and movement cost model used by all pathfinding algorithms.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Set


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Cell = Tuple[int, int]   # (row, col)


class Grid:
    """
    Represents a 2D grid world W ⊂ Z² used as the pathfinding environment.

    Cells are addressed as (row, col) starting at (0, 0) in the top-left
    corner.  Movement is 4-connected (North, South, East, West) with uniform
    unit cost c = 1.0 per step.

    Legend for the internal cell_type matrix:
        0 = free / passable cell
        1 = obstacle / wall (impassable)
    """

    FREE     = 0
    OBSTACLE = 1

    # Cardinal movement directions: (Δrow, Δcol)
    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N, S, W, E

    def __init__(self, rows: int, cols: int):
        """
        Initialise an empty grid of the given dimensions (all cells free).

        Args:
            rows: Number of rows.
            cols: Number of columns.
        """
        self.rows = rows
        self.cols = cols
        # 2D list of cell types: 0 = free, 1 = obstacle
        self._cells: List[List[int]] = [
            [self.FREE] * cols for _ in range(rows)
        ]
        self.start: Optional[Cell] = None
        self.goal:  Optional[Cell] = None

    # ------------------------------------------------------------------
    # Cell accessors
    # ------------------------------------------------------------------

    def in_bounds(self, cell: Cell) -> bool:
        """Return True if *cell* is within the grid boundaries."""
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_obstacle(self, cell: Cell) -> bool:
        """Return True if *cell* is a wall / obstacle."""
        r, c = cell
        return self._cells[r][c] == self.OBSTACLE

    def is_passable(self, cell: Cell) -> bool:
        """Return True if *cell* is in bounds and not an obstacle."""
        return self.in_bounds(cell) and not self.is_obstacle(cell)

    def set_obstacle(self, cell: Cell) -> None:
        """Mark *cell* as an obstacle."""
        r, c = cell
        if self.in_bounds(cell):
            self._cells[r][c] = self.OBSTACLE

    def set_free(self, cell: Cell) -> None:
        """Mark *cell* as free / passable."""
        r, c = cell
        if self.in_bounds(cell):
            self._cells[r][c] = self.FREE

    def get_obstacles(self) -> Set[Cell]:
        """Return a set of all obstacle cells."""
        obstacles: Set[Cell] = set()
        for r in range(self.rows):
            for c in range(self.cols):
                if self._cells[r][c] == self.OBSTACLE:
                    obstacles.add((r, c))
        return obstacles

    # ------------------------------------------------------------------
    # Neighbour generation
    # ------------------------------------------------------------------

    def neighbours(self, cell: Cell) -> List[Cell]:
        """
        Return the list of passable 4-connected neighbours of *cell*.

        Only returns cells that are within bounds AND not obstacles.
        """
        r, c = cell
        result: List[Cell] = []
        for dr, dc in self.DIRECTIONS:
            neighbour = (r + dr, c + dc)
            if self.is_passable(neighbour):
                result.append(neighbour)
        return result

    def move_cost(self, from_cell: Cell, to_cell: Cell) -> float:
        """
        Return the movement cost from *from_cell* to *to_cell*.

        Unit cost model: every valid step costs 1.0.
        """
        return 1.0

    # ------------------------------------------------------------------
    # Heuristic functions
    # ------------------------------------------------------------------

    @staticmethod
    def manhattan(cell: Cell, goal: Cell) -> float:
        """
        Manhattan distance heuristic — admissible for 4-connected grids.

        h(n) = |r_n − r_g| + |c_n − c_g|
        """
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

    @staticmethod
    def euclidean(cell: Cell, goal: Cell) -> float:
        """
        Euclidean (straight-line) distance heuristic.

        h(n) = sqrt((r_n − r_g)² + (c_n − c_g)²)
        """
        return ((cell[0] - goal[0]) ** 2 + (cell[1] - goal[1]) ** 2) ** 0.5

    @staticmethod
    def chebyshev(cell: Cell, goal: Cell) -> float:
        """
        Chebyshev (L∞) distance heuristic — useful for 8-connected grids.

        h(n) = max(|r_n − r_g|, |c_n − c_g|)
        """
        return max(abs(cell[0] - goal[0]), abs(cell[1] - goal[1]))

    # ------------------------------------------------------------------
    # Text visualisation (fallback if matplotlib is unavailable)
    # ------------------------------------------------------------------

    def to_text(
        self,
        path: Optional[List[Cell]] = None,
        explored: Optional[Set[Cell]] = None
    ) -> str:
        """
        Render the grid as a compact ASCII string.

        Legend:
            #  = obstacle
            S  = start
            G  = goal
            *  = path cell
            ·  = explored (visited but not on path)
            .  = free cell
        """
        path_set     = set(path)     if path     else set()
        explored_set = set(explored) if explored else set()

        lines: List[str] = []
        for r in range(self.rows):
            row_chars: List[str] = []
            for c in range(self.cols):
                cell = (r, c)
                if self._cells[r][c] == self.OBSTACLE:
                    row_chars.append("█")
                elif cell == self.start:
                    row_chars.append("S")
                elif cell == self.goal:
                    row_chars.append("G")
                elif cell in path_set:
                    row_chars.append("*")
                elif cell in explored_set:
                    row_chars.append("·")
                else:
                    row_chars.append(".")
            lines.append(" ".join(row_chars))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Grid({self.rows}×{self.cols}, start={self.start}, goal={self.goal})"
