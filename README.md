# 实验室护目镜佩戴识别

机器学习课程实践项目：对**静态头部裁剪头像进行二分类**，只输出佩戴状态，不输出位置框。用户已确认：防护镜正确覆盖眼部为佩戴；普通眼镜、太阳镜、额头佩戴为未佩戴；无法判断的排除。

当前已完成**小样本训练前准备**，尚未训练。180张候选原图复核226个对象，最终保留191张头像：佩戴46、未佩戴145。固定来源组划分为训练133、验证29、测试29，详见[数据处理方案](docs/data_processing_plan.md)和[训练前状态](docs/pretraining_status.json)。

| 集合 | 未佩戴 | 佩戴 | 合计 |
| --- | ---: | ---: | ---: |
| train | 97 | 36 | 133 |
| val | 24 | 5 | 29 |
| test | 24 | 5 | 29 |

所有工厂连续帧保持同组进入训练集；训练集也包含SH17的两类样本。验证、测试使用其他SH17来源组，显式来源组跨集合为0。每个评价集只有5个正例，这是小样本准备版，未证明护目镜识别效果或人员完全独立。

## 当前入口

```powershell
python tests/run_regression.py
python -m src.classification_data check --data dataset/goggles/classification_v2_20261022
python -m src.train_classification --config configs/goggles_classification_v1.yaml --dry-run
```

分类训练入口默认只核查并打印计划，真实训练必须另外显式提供`--start`，且要求当前配置、数据和初始权重已有匹配的预检报告。**本轮没有调用`--start`**，没有梯度更新、训练权重或效果指标。已有`src.train`、`src.evaluate`、`src.predict`是通用检测后端，不用于本次头像分类。

用户最新授权按[分类训练与评价流程](docs/classification_training_workflow.md)推进。已补齐分类专用的`src.evaluate_classification`和`src.predict_classification`，先完成回归，再做2轮短跑；通过后执行既定基线并最后一次测试。当前段落的“尚未训练”描述运行前状态，实际运行完成后更新。

需要重新做无梯度检查时使用已有虚拟环境和新的证据目录：

```powershell
.venv/Scripts/python.exe -X utf8 scripts/preflight_classification.py --config configs/goggles_classification_v1.yaml --output results/preflight_goggles_v2
```

本轮实际使用RTX 3060 Laptop GPU运行16张训练头像的前向，输入16×3×224×224、输出16×2，包含两个类别；参数和缓冲区保持不变，梯度更新0，未构建优化器。189项累计回归通过。训练内存、反向传播、实际训练入口的拟合和最终测试指标未运行。

## 文件与数据

| 位置 | 用途 |
| --- | --- |
| [docs/data_processing_plan.md](docs/data_processing_plan.md) | 已执行的清洗、标签、裁剪、来源组、划分和训练前方案 |
| [docs/pretraining_status.json](docs/pretraining_status.json) | 本轮计数、指纹、实际预检和限制摘要 |
| [configs/goggles_classification_v1.yaml](configs/goggles_classification_v1.yaml) | 当前分类数据、初始权重、训练参数与预检绑定 |
| [docs/data_sources/](docs/data_sources/) | 原始来源、逐对象决定、显式裁剪计划和质量记录 |
| [docs/head_crops.md](docs/head_crops.md) | 通用显式头像裁剪与完整性核查接口 |
| [docs/framework_review.md](docs/framework_review.md) | 当前程序职责与训练边界 |

本地正式数据为`dataset/goggles/classification_v2_20261022/`，对应头像`heads_v2_20261022/`和划分`splits_v2_20261022/`。原图、头像、中间预览、完整运行报告、缓存和权重均被Git忽略；只同步代码、配置、来源哈希与文档。v1试裁剪/初始划分保留为中间审计证据，不是当前训练数据。

本轮无需安装依赖；新增依赖按用户要求仅安装到项目虚拟环境。独立记录见[本轮日志](docs/change_logs/2026-10-22-02-data-preparation-pretraining.md)，版本摘要见[CHANGELOG](CHANGELOG.md)。保留独立提交和初版标签`v0.1.0`，按已有授权普通push同步GitHub。
