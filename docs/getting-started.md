# 快速开始

```bash
cd /Users/glodon/work/test/RL
source .venv_gym/bin/activate

# 交互 UI
streamlit run app.py

# 命令行训练
python training/train_q_learning.py --episodes 500 --use_feedback 1

# 生成数据
python scripts/generate_dataset.py --episodes 50 --policy expert --output data/gridworld_expert.jsonl
```
