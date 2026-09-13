# Task 3: Heuristic Graph Pathfinding Agent

Benchmark generated: 2026-09-13 07:38 UTC

## Summary

This benchmark compares Dijkstra, A* with the Manhattan heuristic, and the bonus tabular Q-learning agent on four reproducible 4-connected grid mazes.
All configurations are generated with fixed seeds and are verified reachable by the generator before search.

## Implemented Components

- `environment/grid.py`: bounded grid, obstacle handling, 4-connected movement, and admissible distance heuristics.
- `environment/generators.py`: Easy, Medium, Hard, and Cul-de-Sac benchmark mazes.
- `algorithms/dijkstra.py`: optimal uniform-cost baseline.
- `algorithms/astar.py`: optimal heuristic-guided search with selectable heuristics.
- `algorithms/q_learning.py`: bonus tabular Q-learning training and greedy path extraction.
- `visualization/plotter.py`: per-maze comparison panels and aggregate metric charts.

## Metrics

`Steps` is the final path length, `Runtime (ms)` includes Q-learning training and inference for the RL row, and `Expanded` is the algorithm-specific explored-node count.

| Configuration | Algorithm | Solved | Steps | Runtime (ms) | Expanded | Path cost |
|---|---|---:|---:|---:|---:|---:|
| Easy (15×15, 10% density) | Dijkstra | Yes | 28 | 1.6694 | 200 | 28.0 |
| Easy (15×15, 10% density) | A* (Manhattan) | Yes | 28 | 0.9907 | 169 | 28.0 |
| Easy (15×15, 10% density) | Q-Learning (RL Agent) | Yes | 28 | 807.9473 | 29 | 28.0 |
| Medium (20×20, 25% density) | Dijkstra | Yes | 52 | 2.9313 | 296 | 52.0 |
| Medium (20×20, 25% density) | A* (Manhattan) | Yes | 52 | 2.7465 | 269 | 52.0 |
| Medium (20×20, 25% density) | Q-Learning (RL Agent) | Yes | 52 | 1923.8360 | 53 | 52.0 |
| Hard (25×25, 35% density) | Dijkstra | Yes | 130 | 7.1828 | 445 | 130.0 |
| Hard (25×25, 35% density) | A* (Manhattan) | Yes | 130 | 6.7323 | 395 | 130.0 |
| Hard (25×25, 35% density) | Q-Learning (RL Agent) | Yes | 130 | 5532.4533 | 131 | 130.0 |
| Cul-de-Sac Trap (20×20, U-shaped) | Dijkstra | Yes | 38 | 5.6548 | 353 | 38.0 |
| Cul-de-Sac Trap (20×20, U-shaped) | A* (Manhattan) | Yes | 38 | 5.4206 | 295 | 38.0 |
| Cul-de-Sac Trap (20×20, U-shaped) | Q-Learning (RL Agent) | Yes | 38 | 1453.3438 | 39 | 38.0 |

## Findings

- Dijkstra and A* produce equal optimal path costs on every solvable benchmark maze.
- A* expands no more nodes than Dijkstra in the tested configurations, while using the goal-directed Manhattan heuristic.
- Q-learning is stochastic model-free training; its reported runtime includes training and greedy extraction, while its expanded count measures inference states only.

## Exported Visualizations

- `artifacts/cul_de_sac_trap_comparison.png`
- `artifacts/easy_comparison.png`
- `artifacts/hard_comparison.png`
- `artifacts/medium_comparison.png`
- `artifacts/metrics_comparison.png`

## Validation

The repository test suite contains 24 unit and integration tests covering grid behavior, heuristic validity, path correctness, optimality, unsolvable grids, and all four benchmark configurations.

## Reproduction

```powershell
.\.venv\Scripts\python.exe benchmark.py
```
