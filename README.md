## 强化学习实战项目：仓储导航 GridWorld（可人类反馈）

本项目通过一个可交互的 GridWorld 仓储导航场景，帮你“知行合一”地学习强化学习（RL）。你可以：

- 了解 RL 的典型适用领域与一个完整场景
- 运行一个自定义环境（仓储导航），训练 Q-learning 智能体
- 通过人类反馈（Like/Dislike）进行奖励塑形，让智能体持续提升
- 生成/下载训练数据（轨迹），用于复现实验或离线分析
- 使用可视化界面（Streamlit）进行交互与训练

---

### 强化学习适用领域（不完整列举）

- 机器人与自动化：路径规划、抓取与放置、协作装配（本项目场景即简化版）
- 运筹与物流：仓储拣选、车辆路径、动态调度与库存补货
- 网络与系统：拥塞控制、自适应缓存、任务调度
- 推荐与广告：顺序决策、长期回报优化
- 游戏与交互：策略学习、对弈与博弈、自适应难度
- 金融与资源分配：组合执行、做市与再平衡、竞价分配

---

### 本项目的完整学习场景：仓储导航 GridWorld

- 环境：一个二维网格代表仓库货架通道，起点在左上角，目标在右下角，存在障碍（如狭窄通道或禁止区域）。
- 任务：智能体（搬运机器人）从起点到达目标，最少步数且避免碰撞。
- 奖励：每步 −1，撞障碍 −5，抵达目标 +10（可通过人类反馈进行奖励塑形）。
- 行动：上/下/左/右 4 个离散动作。
- 人类反馈：你可对智能体在某个状态下的某个动作给予正/负反馈，系统会叠加到奖励上，帮助其更快学会“你认可”的行为。

---

### 快速开始

1) 准备环境

```bash
cd /Users/glodon/work/test/RL
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

2) 运行交互式学习（推荐）

```bash
source .venv/bin/activate
streamlit run app.py
```

交互界面说明：
- 在“交互与反馈”区域：逐步运行一条轨迹，针对“上一步动作”点击 Like/Dislike 记录反馈
- 在“训练”区域：点击“训练若干回合”按钮进行训练并查看奖励曲线
- 在“评估与可视化”区域：查看成功率与 Q 值/策略热力图

3) 命令行离线训练

```bash
source .venv/bin/activate
python training/train_q_learning.py --episodes 500 --use_feedback 1 --render 0
```

4) 生成/下载训练数据

生成本环境的轨迹数据：
```bash
source .venv/bin/activate
python scripts/generate_dataset.py --episodes 50 --policy expert --output data/gridworld_expert.jsonl
```

通用下载器（从任意 URL 下载到 `data/downloads/`）：
```bash
source .venv/bin/activate
python scripts/download_data.py --url https://example.com/some_trajectory.jsonl
```

提示：本项目同时支持“以启代下”的数据生成（expert/heuristic 策略）来代替外部下载，便于离线自给自足。

---

### 目录结构

```
RL/
  app.py                      # Streamlit 交互式界面（人类反馈 + 训练 + 可视化）
  requirements.txt
  rl_project/
    envs/gridworld.py        # 自定义 GridWorld 环境（Gymnasium 接口）
    agents/q_learning.py     # 表格型 Q-learning 智能体
    hitl/feedback_manager.py # 人类反馈存取与奖励塑形
  training/train_q_learning.py
  scripts/
    generate_dataset.py      # 生成轨迹数据（random/expert/policy）
    download_data.py         # 从 URL 下载数据到 data/
  data/                      # 数据与反馈文件存放目录
    README.md
  README.md
```

---

### 设计要点

- 简洁可控：选用表格型 Q-learning 与离散 GridWorld，降低建模与依赖复杂度
- 可交互：Streamlit 便捷采集偏好反馈并即时训练
- 易扩展：环境/智能体/反馈模块解耦，便于替换为更复杂的神经网络或更大规模环境

---

### 常见问题

- 无法启动界面：检查 `pip install -r requirements.txt` 是否成功；或升级 `pip` 再重试
- 训练不收敛：适度增加训练回合、降低 `epsilon`、提高反馈权重 `beta`（见 `feedback_manager.py`）
- 想换环境：参考 `rl_project/envs/gridworld.py` 的 Gymnasium 接口实现，自行新增环境


