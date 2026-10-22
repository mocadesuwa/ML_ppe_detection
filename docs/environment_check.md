# 护目镜项目环境说明

更新日期：2026-10-22。本轮未安装、升级依赖，也未修改系统Python或项目虚拟环境。

| 项目 | 本轮实际检查 |
| --- | --- |
| 解释器 | 项目.venv，Python 3.14.0 |
| PyTorch / Torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| Pillow / PyYAML | 12.3.0 / 6.0.3 |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU，6 GB |
| CUDA | 可用，基础矩阵运算通过 |
| 分类数据与前向 | 16×3×224×224→16×2，两类均加载；eval/inference_mode，无梯度，无参数或缓冲区变化 |
| 前向峰值已分配显存 | 约54.72 MiB，不代表训练显存 |
| 实际训练 | 两轮短跑18次更新；基线11轮99次更新，骨干冻结及最佳检查点重载通过 |
| 训练运行峰值已分配显存 | 两次均约76.53 MiB，包含训练及重载验证，不等于整机GPU占用 |

以上训练次数为第一轮历史记录。第二轮使用同一虚拟环境，未安装依赖：新短跑26次更新，control143次、groups/uniform各182次；四次运行冻结、头更新与最佳权重重载均通过，峰值已分配仍约76.53MiB。当前比较见[改进状态](goggles_improvement_status.json)，完整预检和模型均保存在本地新目录。

完整环境报告保存在本地`results/environment_goggles_classification_20261022.json`，真实无梯度证据为`results/preflight_goggles_v1/report.json`；可同步摘要见[训练前状态](pretraining_status.json)。原图准备和回归使用已有系统Python/Pillow，真实模型前向使用项目虚拟环境。

新增依赖按用户授权仅安装到项目虚拟环境，不安装到系统Python。历史依赖清单见[requirements.txt](../requirements.txt)及[锁定文件](../requirements-lock.txt)；本轮未修改它们。当前模型初始权重来源见[分类初始权重](initial_classification_weights.json)，旧[通用检测权重](initial_weights.json)不用于本次分类。

原无梯度报告保留为训练前证据；本轮另执行了真实反向传播、AdamW更新、早停、最佳权重保存与重载验证，见[训练状态](training_status.json)和[训练报告](goggles_training_report.md)。新增匹配预检分别保存在`results/preflight_goggles_v1_smoke/`及`results/preflight_goggles_v2/`，没有覆盖旧证据。
