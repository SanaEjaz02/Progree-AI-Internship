"""
Tabular Q-Learning Pathfinding Agent — Bonus Reinforcement Learning Loop.

Q-Learning is a MODEL-FREE Reinforcement Learning algorithm that learns an
optimal policy π*(s) through trial-and-error interaction with the environment,
without requiring any prior knowledge of the grid layout.

Unlike A* and Dijkstra which solve the pathfinding problem using graph search
(model-based, single query), Q-Learning:
  - Trains over many episodes, gradually improving a state-action value table Q(s, a).
  - Explores the environment using an ε-greedy exploration strategy.
  - Converges (eventually) to the optimal Q-values and policy under standard
    conditions (α > 0, γ < 1, sufficient exploration).

Q-Update Rule (Temporal Difference Learning):
    Q(s, a) ← Q(s, a) + α · [r + γ · max_{a'} Q(s', a') − Q(s, a)]

where:
    s        = current state (cell)
    a        = action taken (direction of movement)
    s'       = next state after taking action a in state s
    r        = reward received on the transition
    α        = learning rate (step size)
    γ        = discount factor (future reward weight)

Reward Structure:
    +100.0   : reaching the goal cell
    -10.0    : attempting to move into a wall or out of bounds (wasted step)
      -1.0   : each regular free step (encourages efficiency / shorter paths)

After training, the greedy policy (always choosing argmax_a Q(s, a)) is
extracted and followed from start to produce the final path.
"""

from __future__ import annotations
import random
import time
from typing import Dict, List, Tuple, Set, Optional

from environment.grid import Grid, Cell
from .base import SearchResult


# Action space: (Δrow, Δcol) — 4-connected movement
ACTIONS: List[Tuple[int, int]] = [
    (-1, 0),   # North
    ( 1, 0),   # South
    ( 0, -1),  # West
    ( 0,  1),  # East
]
NUM_ACTIONS = len(ACTIONS)


class QLearningAgent:
    """
    Tabular Q-Learning agent that learns to navigate a grid maze.

    Parameters
    ----------
    alpha       : Learning rate α ∈ (0, 1].
    gamma       : Discount factor γ ∈ [0, 1).
    epsilon     : Initial exploration rate ε for ε-greedy policy.
    epsilon_min : Minimum exploration rate after decay.
    epsilon_decay: Multiplicative decay applied to ε after each episode.
    max_episodes: Number of training episodes.
    max_steps   : Maximum steps allowed per episode (prevents infinite loops).
    seed        : Random seed for reproducibility.
    """

    def __init__(
        self,
        alpha:         float = 0.3,
        gamma:         float = 0.95,
        epsilon:       float = 1.0,
        epsilon_min:   float = 0.05,
        epsilon_decay: float = 0.98,
        max_episodes:  int   = 2000,
        max_steps:     int   = 500,
        seed:          int   = 42,
    ):
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.max_episodes  = max_episodes
        self.max_steps     = max_steps
        self.rng           = random.Random(seed)

        # Q-table: maps (cell, action_index) → Q-value
        # Initialised lazily to 0.0 for all unseen state-action pairs
        self.Q: Dict[Tuple[Cell, int], float] = {}

    def _q(self, state: Cell, action: int) -> float:
        """Return Q(state, action), defaulting to 0.0 for unseen pairs."""
        return self.Q.get((state, action), 0.0)

    def _best_action(self, state: Cell) -> int:
        """Return the greedy action: argmax_a Q(state, a)."""
        return max(range(NUM_ACTIONS), key=lambda a: self._q(state, a))

    def _choose_action(self, state: Cell) -> int:
        """
        ε-greedy action selection:
          - With probability ε: pick a random action (exploration).
          - With probability 1-ε: pick the greedy action (exploitation).
        """
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, NUM_ACTIONS - 1)
        return self._best_action(state)

    def _step(self, grid: Grid, state: Cell, action_idx: int) -> Tuple[Cell, float, bool]:
        """
        Apply *action_idx* in *state*, return (next_state, reward, done).

        Reward shaping:
            +100.0 for reaching the goal.
            -10.0  for colliding with a wall or grid boundary.
             -1.0  for each regular free step.
        """
        dr, dc = ACTIONS[action_idx]
        next_cell = (state[0] + dr, state[1] + dc)

        if not grid.is_passable(next_cell):
            # Collision: stay in place, penalise
            return state, -10.0, False

        if next_cell == grid.goal:
            return next_cell, 100.0, True

        return next_cell, -1.0, False

    def train(self, grid: Grid) -> None:
        """
        Run the Q-Learning training loop on *grid*.

        At each episode:
          1. Reset agent to grid.start.
          2. Choose actions via ε-greedy, observe reward and next state.
          3. Update Q-table via the Bellman temporal-difference equation.
          4. Decay ε toward epsilon_min.
        """
        for _ in range(self.max_episodes):
            state = grid.start
            for _ in range(self.max_steps):
                # 1. Select action using current ε-greedy policy
                action = self._choose_action(state)

                # 2. Execute action in the environment
                next_state, reward, done = self._step(grid, state, action)

                # 3. TD update: Q(s, a) ← Q(s, a) + α[r + γ max Q(s', ·) − Q(s, a)]
                best_next_q = max(self._q(next_state, a) for a in range(NUM_ACTIONS))
                old_q       = self._q(state, action)
                new_q       = old_q + self.alpha * (
                    reward + self.gamma * best_next_q - old_q
                )
                self.Q[(state, action)] = new_q

                state = next_state
                if done:
                    break

            # 4. Decay exploration rate
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def extract_path(self, grid: Grid) -> SearchResult:
        """
        After training, follow the greedy policy from grid.start to grid.goal
        and return a SearchResult.

        The inference run itself is timed and its step count and explored set
        are recorded.  The Q-table training time is NOT included.
        """
        start = grid.start
        goal  = grid.goal
        t0    = time.perf_counter()

        state    = start
        path     = [state]
        explored: Set[Cell] = {state}
        visited  = {state}
        steps    = 0
        max_inf_steps = grid.rows * grid.cols * 2  # hard cap

        while state != goal and steps < max_inf_steps:
            action     = self._best_action(state)
            dr, dc     = ACTIONS[action]
            next_cell  = (state[0] + dr, state[1] + dc)

            if not grid.is_passable(next_cell) or next_cell in visited:
                # Agent is stuck or looping — abort
                break

            state = next_cell
            path.append(state)
            explored.add(state)
            visited.add(state)
            steps += 1

        success = (state == goal)
        elapsed = (time.perf_counter() - t0) * 1000.0

        return SearchResult(
            algorithm  = "Q-Learning (RL Agent)",
            success    = success,
            path       = path if success else [],
            path_cost  = float(steps),
            steps      = steps,
            runtime_ms = elapsed,
            expanded   = len(explored),   # states visited during inference
            explored   = explored,
        )


def run_q_learning(grid: Grid, **kwargs) -> Tuple[SearchResult, float]:
    """
    Train a Q-Learning agent on *grid* and return its greedy path.

    Returns:
        (SearchResult, training_time_ms): result from inference + training time.
    """
    agent = QLearningAgent(**kwargs)

    t_train = time.perf_counter()
    agent.train(grid)
    train_ms = (time.perf_counter() - t_train) * 1000.0

    result = agent.extract_path(grid)
    return result, train_ms
