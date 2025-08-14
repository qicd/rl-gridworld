from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


StateAction = Tuple[int, int]


@dataclass
class FeedbackConfig:
    file_path: Path
    beta: float = 0.5  # weight added to environment reward


class FeedbackManager:
    """Manage human feedback for state-action pairs and shape rewards.

    Feedback is accumulated as integer scores per (state, action). The shaped
    reward is `env_reward + beta * score`.
    """
    def __init__(self, config: FeedbackConfig) -> None:
        self.config = config
        self.file_path = config.file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._feedback: Dict[str, int] = {}
        self._load()

    def _key(self, state: int, action: int) -> str:
        return f"{state}:{action}"

    def add_feedback(self, state: int, action: int, label: int) -> None:
        # label: +1 like, -1 dislike
        key = self._key(state, action)
        self._feedback[key] = self._feedback.get(key, 0) + int(label)
        self._save()

    def get_feedback_score(self, state: int, action: int) -> float:
        key = self._key(state, action)
        return float(self._feedback.get(key, 0))

    def shaped_reward(self, env_reward: float, state: int, action: int) -> float:
        return float(env_reward + self.config.beta * self.get_feedback_score(state, action))

    def _load(self) -> None:
        if self.file_path.exists():
            try:
                data = json.loads(self.file_path.read_text())
                if isinstance(data, dict):
                    self._feedback = {str(k): int(v) for k, v in data.items()}
            except Exception:
                pass

    def _save(self) -> None:
        try:
            self.file_path.write_text(json.dumps(self._feedback, ensure_ascii=False, indent=2))
        except Exception:
            pass


