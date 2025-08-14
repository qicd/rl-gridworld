from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from rl_project.envs import GridWorldEnv
from rl_project.agents import QLearningAgent


def run_policy(env: GridWorldEnv, policy: str, agent: QLearningAgent | None) -> Dict:
    state, _ = env.reset()
    done = False
    episode = {"transitions": []}
    while not done:
        if policy == "random":
            action = int(env.action_space.sample())
        elif policy == "expert":
            # simple greedy heuristic: move closer to goal by Manhattan distance
            action = greedy_towards_goal(env, state)
        elif policy == "agent":
            assert agent is not None
            action = agent.greedy_action(state)
        else:
            raise ValueError("Unknown policy")
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode["transitions"].append({
            "s": int(state), "a": int(action), "r": float(reward), "s_next": int(next_state), "done": bool(done)
        })
        state = next_state
    return episode


def greedy_towards_goal(env: GridWorldEnv, state: int) -> int:
    # convert to position
    y, x = divmod(state, env.grid.width)
    sx, sy = x, y
    gx, gy = env.grid.goal
    # Try to reduce manhattan distance, check obstacle collision by simulating move
    candidates = []
    # up
    candidates.append((0, (sx, max(0, sy - 1))))
    # down
    candidates.append((1, (sx, min(env.grid.height - 1, sy + 1))))
    # left
    candidates.append((2, (max(0, sx - 1), sy)))
    # right
    candidates.append((3, (min(env.grid.width - 1, sx + 1), sy)))

    def manhattan(p):
        return abs(p[0] - gx) + abs(p[1] - gy)

    # Avoid obstacles by assigning big distance
    scored = []
    for a, (nx, ny) in candidates:
        if (nx, ny) in env.grid.obstacles:
            scored.append((a, 1e9))
        else:
            scored.append((a, manhattan((nx, ny))))
    scored.sort(key=lambda x: x[1])
    return int(scored[0][0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--policy", type=str, default="expert", choices=["random", "expert", "agent"])
    parser.add_argument("--output", type=str, default="data/gridworld_dataset.jsonl")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    env = GridWorldEnv()
    agent = None
    if args.policy == "agent":
        # Quickly pre-train an agent with a small number of episodes
        agent = QLearningAgent(env.observation_space.n, env.action_space.n)
        for _ in range(200):
            s, _ = env.reset(seed=rng.integers(0, 1_000_000))
            done = False
            while not done:
                a = agent.select_action(s)
                s2, r, term, trunc, _ = env.step(a)
                agent.update(s, a, r, s2, term or trunc)
                s = s2
                done = term or trunc

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for _ in range(args.episodes):
            ep = run_policy(env, args.policy, agent)
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")
    print(f"Saved dataset to {out}")


if __name__ == "__main__":
    main()


