from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Ensure project root on sys.path for `streamlit run app.py`
import sys
from pathlib import Path as _Path
_project_root = str(_Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rl_project.envs import GridWorldEnv
from rl_project.agents import QLearningAgent, QLearningConfig
from rl_project.hitl.feedback_manager import FeedbackManager, FeedbackConfig


@st.cache_resource
def get_env_and_agent():
    env = GridWorldEnv()
    agent = QLearningAgent(env.observation_space.n, env.action_space.n, QLearningConfig())
    feedback = FeedbackManager(FeedbackConfig(file_path=Path("data/feedback/gridworld_feedback.json")))
    return env, agent, feedback


def run_one_step(env: GridWorldEnv, agent: QLearningAgent, state: int, use_feedback: bool, feedback_mgr: FeedbackManager):
    action = agent.select_action(state)
    next_state, reward, terminated, truncated, _ = env.step(action)
    if use_feedback:
        reward = feedback_mgr.shaped_reward(reward, state, action)
    agent.update(state, action, reward, next_state, terminated or truncated)
    return action, next_state, reward, terminated or truncated


def draw_grid(env: GridWorldEnv):
    arr = env.as_array()
    fig, ax = plt.subplots(figsize=(1, 1))
    cmap = plt.get_cmap("coolwarm")
    ax.imshow(arr, cmap=cmap, vmin=-1, vmax=2)
    ax.set_xticks(range(env.grid.width))
    ax.set_yticks(range(env.grid.height))
    ax.grid(True, color="#cccccc", linewidth=0.5)
    st.pyplot(fig)


def main():
    env, agent, feedback_mgr = get_env_and_agent()
    use_feedback = st.sidebar.checkbox("启用人类反馈奖励塑形", value=True)
    draw_grid(env)

    # Interact
    st.subheader("交互与反馈")
    if "state" not in st.session_state:
        s, _ = env.reset()
        st.session_state.state = s
        st.session_state.last_action = None
        st.session_state.done = False

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("执行一步") and not st.session_state.done:
            a, s2, r, d = run_one_step(env, agent, st.session_state.state, use_feedback, feedback_mgr)
            st.session_state.last_action = a
            st.session_state.state = s2
            st.session_state.done = d
    with col2:
        if st.button("重置"):
            s, _ = env.reset()
            st.session_state.state = s
            st.session_state.last_action = None
            st.session_state.done = False
    with col3:
        st.write("当前状态:", st.session_state.state)

    if st.session_state.last_action is not None:
        st.write("对上一步动作给出反馈：")
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            if st.button("👍 Like"):
                feedback_mgr.add_feedback(st.session_state.state, int(st.session_state.last_action), +1)
        with fcol2:
            if st.button("👎 Dislike"):
                feedback_mgr.add_feedback(st.session_state.state, int(st.session_state.last_action), -1)

    # Training
    st.subheader("训练")
    episodes = st.number_input("训练回合数", min_value=10, max_value=5000, value=200, step=10)
    if st.button("训练若干回合"):
        returns: List[float] = []
        for _ in range(int(episodes)):
            s, _ = env.reset()
            done = False
            ep_ret = 0.0
            while not done:
                a = agent.select_action(s)
                s2, r, term, trunc, _ = env.step(a)
                if use_feedback:
                    r = feedback_mgr.shaped_reward(r, s, a)
                agent.update(s, a, r, s2, term or trunc)
                ep_ret += r
                s = s2
                done = term or trunc
            returns.append(ep_ret)
        st.line_chart(returns)
        st.success(f"训练完成，最近50回合平均回报：{np.mean(returns[-50:]) if len(returns)>=50 else np.mean(returns):.2f}")

    # Evaluation
    st.subheader("评估与可视化")
    if st.button("显示当前策略 Q 值最大动作图"):
        policy = np.argmax(agent.q_table, axis=1).reshape(env.grid.height, env.grid.width)
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(policy, cmap="Accent", vmin=0, vmax=3)
        ax.set_title("Greedy Policy (0=上,1=下,2=左,3=右)")
        st.pyplot(fig)


if __name__ == "__main__":
    main()


