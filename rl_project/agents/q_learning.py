from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class QLearningConfig:
    learning_rate: float = 0.1
    discount_gamma: float = 0.99
    epsilon_start: float = 0.2
    epsilon_final: float = 0.05
    epsilon_decay_steps: int = 10_000


class QLearningAgent:
    """Tabular Q-learning agent with epsilon-greedy policy.

    Args:
      num_states: Size of discrete observation space.
      num_actions: Size of discrete action space.
      config: Hyperparameters (lr, gamma, epsilon schedule).
    """
    def __init__(self, num_states: int, num_actions: int, config: Optional[QLearningConfig] = None) -> None:
        self.num_states = num_states
        self.num_actions = num_actions
        self.config = config or QLearningConfig()

        self.q_table = np.zeros((num_states, num_actions), dtype=np.float32)
        self._steps_done = 0

    def select_action(self, state: int) -> int:
        epsilon = self._current_epsilon()
        if np.random.rand() < epsilon:
            return int(np.random.randint(self.num_actions))
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        best_next = 0.0 if done else float(np.max(self.q_table[next_state]))
        target = reward + self.config.discount_gamma * best_next
        td_error = target - float(self.q_table[state, action])
        self.q_table[state, action] += self.config.learning_rate * td_error
        self._steps_done += 1

    def _current_epsilon(self) -> float:
        frac = min(1.0, self._steps_done / max(1, self.config.epsilon_decay_steps))
        return float(self.config.epsilon_start + frac * (self.config.epsilon_final - self.config.epsilon_start))

    def greedy_action(self, state: int) -> int:
        return int(np.argmax(self.q_table[state]))


