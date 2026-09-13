"""
Base Search Infrastructure: Abstract Algorithm Interface & SearchResult Container.

Defines the standardised data container returned by every pathfinding
algorithm and the abstract base class all algorithms must implement.
"""

from __future__ import annotations
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Set

from environment.grid import Grid, Cell


@dataclass
class SearchResult:
    """
    Standardised container for all outputs produced by a pathfinding run.

    Attributes:
        algorithm   : Human-readable name of the algorithm (e.g. 'A* (Manhattan)').
        success     : True if the goal was reached.
        path        : Ordered list of cells from start to goal (inclusive).
                      Empty list when no path was found.
        path_cost   : Total cumulative movement cost along *path*.
        steps       : Number of grid steps taken (= len(path) - 1 if path found).
        runtime_ms  : Wall-clock execution time in milliseconds.
        expanded    : Total number of nodes popped from the priority queue and
                      evaluated (the primary measure of algorithm efficiency).
        explored    : Set of all cells that were placed on the open/closed list.
                      Used for visualising the search frontier heatmap.
    """
    algorithm:   str       = ""
    success:     bool      = False
    path:        List[Cell] = field(default_factory=list)
    path_cost:   float     = 0.0
    steps:       int       = 0
    runtime_ms:  float     = 0.0
    expanded:    int       = 0
    explored:    Set[Cell] = field(default_factory=set)

    def summary(self) -> str:
        """Return a compact one-line performance summary string."""
        if self.success:
            return (
                f"[{self.algorithm}] SUCCESS | "
                f"Steps: {self.steps:4d} | "
                f"Cost: {self.path_cost:6.1f} | "
                f"Expanded: {self.expanded:5d} nodes | "
                f"Time: {self.runtime_ms:6.2f} ms"
            )
        return f"[{self.algorithm}] NO PATH FOUND | Time: {self.runtime_ms:.2f} ms"


class PathfindingAlgorithm(ABC):
    """
    Abstract base class for all pathfinding algorithms.

    Subclasses must implement the `search` method which receives a configured
    Grid and returns a SearchResult.
    """

    def __init__(self, name: str):
        """
        Args:
            name: Human-readable algorithm label used in logs and reports.
        """
        self.name = name

    @abstractmethod
    def search(self, grid: Grid) -> SearchResult:
        """
        Execute the pathfinding search on *grid* from grid.start to grid.goal.

        Args:
            grid: The fully configured Grid with start and goal set.

        Returns:
            SearchResult containing path, metrics, and explored-node set.
        """
        ...

    @staticmethod
    def _reconstruct_path(came_from: dict, current: Cell) -> List[Cell]:
        """
        Trace back through the *came_from* predecessor map from *current*
        to the start node to reconstruct the full path.

        Args:
            came_from: Mapping from each cell to the cell it was reached from.
            current  : The goal cell from which to start backtracking.

        Returns:
            Ordered list [start, ..., goal].
        """
        path: List[Cell] = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)   # append the start node
        path.reverse()
        return path

    def _make_result(
        self,
        success: bool,
        came_from: dict,
        goal: Cell,
        g_cost: float,
        expanded: int,
        explored: Set[Cell],
        start_time: float
    ) -> SearchResult:
        """
        Construct a SearchResult from the finished search state.
        Handles both successful and failed searches.
        """
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        if success:
            path = self._reconstruct_path(came_from, goal)
            return SearchResult(
                algorithm  = self.name,
                success    = True,
                path       = path,
                path_cost  = g_cost,
                steps      = len(path) - 1,
                runtime_ms = elapsed_ms,
                expanded   = expanded,
                explored   = explored,
            )
        return SearchResult(
            algorithm  = self.name,
            success    = False,
            runtime_ms = elapsed_ms,
            expanded   = expanded,
            explored   = explored,
        )
