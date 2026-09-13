"""
Unit & Integration Tests for Task 3: Heuristic Graph Pathfinding Agent.

Tests verify:
  1. A* and Dijkstra find identical optimal path costs on solvable grids.
  2. A* expands ≤ nodes than Dijkstra (heuristic pruning property).
  3. Both algorithms correctly detect unsolvable mazes.
  4. Grid bounds checking and obstacle collision logic.
  5. Path validity — every consecutive step must be a valid move.
  6. All 4 benchmark maze configurations are solvable by A* and Dijkstra.
"""

import sys
import os
import unittest

# Ensure project root is on sys.path so that imports work from tests/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.grid import Grid
from environment.generators import (
    make_easy_grid, make_medium_grid, make_hard_grid, make_cul_de_sac_grid
)
from algorithms.dijkstra import Dijkstra
from algorithms.astar    import AStar


class TestGridEnvironment(unittest.TestCase):
    """Tests for the Grid environment and its methods."""

    def setUp(self):
        self.grid = Grid(5, 5)
        self.grid.start = (0, 0)
        self.grid.goal  = (4, 4)

    def test_in_bounds(self):
        self.assertTrue(self.grid.in_bounds((0, 0)))
        self.assertTrue(self.grid.in_bounds((4, 4)))
        self.assertFalse(self.grid.in_bounds((-1, 0)))
        self.assertFalse(self.grid.in_bounds((5, 0)))
        self.assertFalse(self.grid.in_bounds((0, 5)))

    def test_obstacle_set_and_query(self):
        self.assertFalse(self.grid.is_obstacle((2, 2)))
        self.grid.set_obstacle((2, 2))
        self.assertTrue(self.grid.is_obstacle((2, 2)))
        self.grid.set_free((2, 2))
        self.assertFalse(self.grid.is_obstacle((2, 2)))

    def test_neighbours_open_grid(self):
        # Centre cell (2,2) in empty 5x5 should have 4 neighbours
        nbrs = self.grid.neighbours((2, 2))
        self.assertEqual(len(nbrs), 4)

    def test_neighbours_corner(self):
        # Corner cell (0,0) should have 2 neighbours
        nbrs = self.grid.neighbours((0, 0))
        self.assertEqual(len(nbrs), 2)
        self.assertIn((0, 1), nbrs)
        self.assertIn((1, 0), nbrs)

    def test_neighbours_blocked(self):
        # Block all neighbours of (1,1)
        for n in [(0, 1), (2, 1), (1, 0), (1, 2)]:
            self.grid.set_obstacle(n)
        nbrs = self.grid.neighbours((1, 1))
        self.assertEqual(nbrs, [])

    def test_heuristics_non_negative(self):
        for cell in [(0, 0), (2, 3), (4, 4)]:
            goal = (4, 4)
            self.assertGreaterEqual(Grid.manhattan(cell, goal), 0)
            self.assertGreaterEqual(Grid.euclidean(cell, goal), 0)
            self.assertGreaterEqual(Grid.chebyshev(cell, goal), 0)

    def test_heuristics_zero_at_goal(self):
        goal = (4, 4)
        self.assertEqual(Grid.manhattan(goal, goal), 0)
        self.assertEqual(Grid.euclidean(goal, goal), 0)
        self.assertEqual(Grid.chebyshev(goal, goal), 0)


class TestDijkstra(unittest.TestCase):
    """Tests for Dijkstra's algorithm."""

    def _simple_grid(self):
        g = Grid(5, 5)
        g.start = (0, 0)
        g.goal  = (4, 4)
        return g

    def test_finds_path_open_grid(self):
        result = Dijkstra().search(self._simple_grid())
        self.assertTrue(result.success)
        self.assertGreater(result.steps, 0)
        self.assertEqual(result.path[0],  (0, 0))
        self.assertEqual(result.path[-1], (4, 4))

    def test_optimal_path_cost_open_grid(self):
        # On a 5×5 open grid, shortest Manhattan path = 8 steps
        result = Dijkstra().search(self._simple_grid())
        self.assertEqual(result.steps, 8)

    def test_detects_unsolvable(self):
        g = Grid(5, 5)
        g.start = (0, 0)
        g.goal  = (4, 4)
        # Wall off the goal completely
        for r in range(5):
            g.set_obstacle((r, 3))
        result = Dijkstra().search(g)
        self.assertFalse(result.success)
        self.assertEqual(result.path, [])

    def test_path_validity(self):
        result = Dijkstra().search(self._simple_grid())
        path = result.path
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]
            step = abs(r1 - r2) + abs(c1 - c2)
            self.assertEqual(step, 1, f"Invalid step at index {i}: {path[i]} → {path[i+1]}")


class TestAStar(unittest.TestCase):
    """Tests for A* Search with different heuristics."""

    def _simple_grid(self):
        g = Grid(5, 5)
        g.start = (0, 0)
        g.goal  = (4, 4)
        return g

    def test_finds_path_manhattan(self):
        result = AStar("manhattan").search(self._simple_grid())
        self.assertTrue(result.success)
        self.assertEqual(result.path[0],  (0, 0))
        self.assertEqual(result.path[-1], (4, 4))

    def test_finds_path_euclidean(self):
        result = AStar("euclidean").search(self._simple_grid())
        self.assertTrue(result.success)

    def test_finds_path_chebyshev(self):
        result = AStar("chebyshev").search(self._simple_grid())
        self.assertTrue(result.success)

    def test_optimal_cost_equals_dijkstra(self):
        g   = self._simple_grid()
        d   = Dijkstra().search(g)
        a   = AStar("manhattan").search(g)
        self.assertEqual(d.steps, a.steps)
        self.assertAlmostEqual(d.path_cost, a.path_cost, places=5)

    def test_astar_expands_fewer_or_equal_nodes(self):
        g = self._simple_grid()
        d = Dijkstra().search(g)
        a = AStar("manhattan").search(g)
        self.assertLessEqual(
            a.expanded, d.expanded,
            msg=(
                f"A* should expand ≤ nodes as Dijkstra. "
                f"A*={a.expanded}, Dijkstra={d.expanded}"
            )
        )

    def test_detects_unsolvable(self):
        g = Grid(5, 5)
        g.start = (0, 0)
        g.goal  = (4, 4)
        for r in range(5):
            g.set_obstacle((r, 3))
        result = AStar().search(g)
        self.assertFalse(result.success)

    def test_invalid_heuristic_raises(self):
        with self.assertRaises(ValueError):
            AStar("bogus_heuristic")

    def test_path_validity(self):
        result = AStar("manhattan").search(self._simple_grid())
        path = result.path
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]
            step = abs(r1 - r2) + abs(c1 - c2)
            self.assertEqual(step, 1)


class TestAlgorithmVsAlgorithm(unittest.TestCase):
    """Cross-algorithm correctness and efficiency tests on benchmark mazes."""

    def _run_both(self, grid):
        d = Dijkstra().search(grid)
        a = AStar("manhattan").search(grid)
        return d, a

    def test_easy_grid_both_solve(self):
        grid = make_easy_grid()
        d, a = self._run_both(grid)
        self.assertTrue(d.success, "Dijkstra must solve Easy grid")
        self.assertTrue(a.success, "A* must solve Easy grid")
        self.assertEqual(d.steps, a.steps)

    def test_medium_grid_both_solve(self):
        grid = make_medium_grid()
        d, a = self._run_both(grid)
        self.assertTrue(d.success, "Dijkstra must solve Medium grid")
        self.assertTrue(a.success, "A* must solve Medium grid")
        self.assertEqual(d.steps, a.steps)

    def test_hard_grid_both_solve(self):
        grid = make_hard_grid()
        d, a = self._run_both(grid)
        self.assertTrue(d.success, "Dijkstra must solve Hard grid")
        self.assertTrue(a.success, "A* must solve Hard grid")
        self.assertEqual(d.steps, a.steps)

    def test_cul_de_sac_both_solve(self):
        grid = make_cul_de_sac_grid()
        d, a = self._run_both(grid)
        self.assertTrue(d.success, "Dijkstra must solve Cul-de-Sac grid")
        self.assertTrue(a.success, "A* must solve Cul-de-Sac grid")
        self.assertEqual(d.steps, a.steps)

    def test_astar_always_efficient(self):
        """A* must expand ≤ nodes than Dijkstra on all 4 configurations."""
        for make_fn in [make_easy_grid, make_medium_grid, make_hard_grid, make_cul_de_sac_grid]:
            grid = make_fn()
            d, a = self._run_both(grid)
            self.assertLessEqual(
                a.expanded, d.expanded,
                msg=f"A* expanded more nodes than Dijkstra on {make_fn.__name__}"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
