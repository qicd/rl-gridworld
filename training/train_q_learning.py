from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

from rl_project.envs import GridWorldEnv
from rl_project.agents import QLearningAgent, QLearningConfig
from rl_project.hitl.feedback_manager import FeedbackManager, FeedbackConfig


def run_training(episodes: int, use_feedback: bool, render: bool, seed: int) -> Tuple[QLearningAgent, List[float]]:
    env = GridWorldEnv()
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    agent = QLearningAgent(env.observation_space.n, env.action_space.n, QLearningConfig())

    feedback_mgr = None
    if use_feedback:
        feedback_file = Path("data/feedback/gridworld_feedback.json")
        feedback_mgr = FeedbackManager(FeedbackConfig(file_path=feedback_file))

    episode_returns: List[float] = []
    for ep in range(episodes):
        state, _ = env.reset(seed=rng.integers(0, 1_000_000))
        ep_return = 0.0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if use_feedback and feedback_mgr is not None:
                reward = feedback_mgr.shaped_reward(reward, state, action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            ep_return += reward
            if render and ep % 50 == 0:
                print(env.render())
                print("----")
        episode_returns.append(ep_return)
    return agent, episode_returns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--use_feedback", type=int, default=0)
    parser.add_argument("--render", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    agent, returns = run_training(args.episodes, bool(args.use_feedback), bool(args.render), args.seed)
    print(f"Training finished. Mean return(last 50): {np.mean(returns[-50:]) if len(returns)>=50 else np.mean(returns):.2f}")


if __name__ == "__main__":
    main()


