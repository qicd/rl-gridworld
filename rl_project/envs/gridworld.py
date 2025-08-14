from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from rl_project.spaces import Discrete


Action = int  # 0: up, 1: down, 2: left, 3: right
Position = Tuple[int, int]


@dataclass
class GridSpec:
    width: int
    height: int
    start: Position
    goal: Position
    obstacles: List[Position]


def make_default_grid() -> GridSpec:
    # 5x5 grid with a few obstacles
    return GridSpec(
        width=5,
        height=5,
        start=(0, 0),
        goal=(4, 4),
        obstacles=[(1, 2), (2, 2), (3, 1)],
    )


class GridWorldEnv:
    metadata = {"render_modes": ["ansi"], "render_fps": 10}

    def __init__(
        self,
        grid: Optional[GridSpec] = None,
        step_penalty: float = -1.0,
        obstacle_penalty: float = -5.0,
        goal_reward: float = 10.0,
        render_mode: Optional[str] = None,
    ) -> None:
        self.grid = grid or make_default_grid()
        self.step_penalty = step_penalty
        self.obstacle_penalty = obstacle_penalty
        self.goal_reward = goal_reward
        self.render_mode = render_mode

        self.observation_space = Discrete(self.grid.width * self.grid.height)
        self.action_space = Discrete(4)

        self._position: Position = self.grid.start
        self._terminated = False
        self._truncated = False

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            np.random.seed(seed)
        self._position = self.grid.start
        self._terminated = False
        self._truncated = False
        return self._pos_to_state(self._position), {}

    # Gymnasium step signature
    def step(self, action: Action):
        if self._terminated or self._truncated:
            raise RuntimeError("Call reset() before step() after termination.")

        next_position = self._move(self._position, action)
        reward = self.step_penalty
        info: Dict = {}

        if next_position in self.grid.obstacles:
            # collision -> stay put and penalize
            reward += self.obstacle_penalty
            next_position = self._position
            info["collision"] = True

        self._position = next_position

        if self._position == self.grid.goal:
            reward += self.goal_reward
            self._terminated = True

        observation = self._pos_to_state(self._position)
        return observation, float(reward), self._terminated, self._truncated, info

    def _move(self, pos: Position, action: Action) -> Position:
        x, y = pos
        if action == 0:  # up
            y = max(0, y - 1)
        elif action == 1:  # down
            y = min(self.grid.height - 1, y + 1)
        elif action == 2:  # left
            x = max(0, x - 1)
        elif action == 3:  # right
            x = min(self.grid.width - 1, x + 1)
        else:
            raise ValueError(f"Invalid action: {action}")
        return (x, y)

    def _pos_to_state(self, pos: Position) -> int:
        x, y = pos
        return y * self.grid.width + x

    def _state_to_pos(self, state: int) -> Position:
        y, x = divmod(state, self.grid.width)
        return (x, y)

    # Simple ANSI rendering for console
    def render(self):
        grid = np.full((self.grid.height, self.grid.width), fill_value=".", dtype=object)
        for (ox, oy) in self.grid.obstacles:
            grid[oy, ox] = "#"
        sx, sy = self.grid.start
        gx, gy = self.grid.goal
        grid[sy, sx] = "S"
        grid[gy, gx] = "G"
        ax, ay = self._position
        grid[ay, ax] = "A"
        lines = [" ".join(map(str, row)) for row in grid]
        return "\n".join(lines)

    # Utility for visualization
    def as_array(self, agent_pos: Optional[Position] = None) -> np.ndarray:
        arr = np.zeros((self.grid.height, self.grid.width), dtype=int)
        for (ox, oy) in self.grid.obstacles:
            arr[oy, ox] = -1
        gx, gy = self.grid.goal
        arr[gy, gx] = 2
        if agent_pos is None:
            agent_pos = self._position
        ax, ay = agent_pos
        arr[ay, ax] = 1
        return arr


