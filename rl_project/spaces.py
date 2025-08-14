from __future__ import annotations

import random


class Discrete:
    def __init__(self, n: int) -> None:
        assert n > 0
        self.n = int(n)

    def sample(self) -> int:
        return int(random.randrange(self.n))


