# 护目镜项目环境说明

更新日期：2026-10-22。本轮使用已有系统Python的标准库与Pillow获取并核查候选原图，没有安装依赖、检查GPU或运行任务模型。2026-10-14的回归是历史验证，不是本轮重跑。

- 通用数据和回归主要使用已有Pillow、PyYAML及Python标准库。
- 模型后端仍包含PyTorch、Ultralytics等依赖；最终任务明确后再核对所需模型与环境。
- 依赖要求见[requirements.txt](../requirements.txt)，历史精确锁定见[requirements-lock.txt](../requirements-lock.txt)。锁定文件不代表本轮已复查模型运行环境。
- 已有通用初始权重来源记录见[initial_weights.json](initial_weights.json)，不代表护目镜训练权重或最终选型。
- 用户已授权按需安装依赖；新增依赖仅装入项目虚拟环境，不安装到系统Python。本轮已有依赖够用，未创建、升级或修改环境。

当前可查看入口帮助及运行累计回归：

```powershell
python tests/run_regression.py
python -m src.environment --help
python scripts/preflight_training.py --help
```

无梯度前向不能证明训练显存、反向传播或护目镜效果。本轮仅推进小规模数据集检索、获取及核查；模型检查和训练未运行。
