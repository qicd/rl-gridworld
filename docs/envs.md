# 环境（GridWorld）

- 状态：网格坐标，编码为单个离散整数（`width * y + x`）
- 动作：上/下/左/右（0/1/2/3）
- 奖励：每步 −1，撞障碍额外 −5，抵达目标 +10
- 终止：到达目标

对应实现：`rl_project/envs/gridworld.py`
