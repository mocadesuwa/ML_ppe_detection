# 护目镜项目环境说明

更新日期：2026-10-14。本轮沿用系统Python已有依赖运行合成样本回归，没有安装依赖、重新检查GPU或运行任务模型。历史安装过程保留在独立日志中。

- 通用数据和回归主要使用已有Pillow、PyYAML及Python标准库。
- 模型后端仍包含PyTorch、Ultralytics等依赖；最终任务明确后再核对所需模型与环境。
- 依赖要求见[requirements.txt](../requirements.txt)，历史精确锁定见[requirements-lock.txt](../requirements-lock.txt)。锁定文件不代表本轮已复查模型运行环境。
- 已有通用初始权重来源记录见[initial_weights.json](initial_weights.json)，不代表护目镜训练权重或最终选型。

当前可查看入口帮助及运行累计回归：

```powershell
python tests/run_regression.py
python -m src.environment --help
python scripts/preflight_training.py --help
```

无梯度前向不能证明训练显存、反向传播或护目镜效果。真实模型检查、训练和依赖安装按用户明确推进执行；本轮未运行。
