# 实验室护目镜佩戴识别

机器学习课程实践项目：对**静态头部裁剪头像进行二分类**，只输出佩戴状态，不输出位置框。用户已确认：防护镜正确覆盖眼部为佩戴；普通眼镜、太阳镜、额头佩戴为未佩戴；无法判断的排除。

当前已完成**数据准备、两轮短跑、首版基线训练和一次冻结测试**。测试Accuracy为62.07%、macro-F1为0.5054，佩戴召回率40%；效果较弱，仍是小样本基线。详见[训练与评价报告](docs/goggles_training_report.md)和[运行摘要](docs/training_status.json)。

用户已授权继续改进。本轮新增110张已复核头像，建立272张v3实验数据（旧29张测试封存），比较数据扩充、来源组抽样及类别加权；选择和评价规则预先锁定，见[改进方案](docs/goggles_improvement_plan.md)。新实验结果完成后另行报告。

180张候选原图复核226个对象，最终保留191张头像：佩戴46、未佩戴145。数据和来源组划分保持冻结，详见[数据处理方案](docs/data_processing_plan.md)。

| 集合 | 未佩戴 | 佩戴 | 合计 |
| --- | ---: | ---: | ---: |
| train | 97 | 36 | 133 |
| val | 24 | 5 | 29 |
| test | 24 | 5 | 29 |

所有工厂连续帧保持同组进入训练集；训练集也包含SH17的两类样本。验证、测试使用其他SH17来源组，显式来源组跨集合为0。训练133张仅12个来源组，佩戴样本覆盖3组；每个评价集只有5个正例，尚无实际实验室验证或独立复标。

## 当前入口

```powershell
python tests/run_regression.py
python -m src.classification_data check --data dataset/goggles/classification_v2_20261022
python -m src.train_classification --config configs/goggles_classification_v1.yaml --dry-run
```

分类训练入口默认只核查并打印计划，真实训练必须显式提供`--start`，且要求配置、数据和初始权重已有匹配预检。本轮按[确认流程](docs/classification_training_workflow.md)先短跑2轮18次更新，通过骨干冻结、分类头更新和最佳权重重载检查后，独立开始基线；基线第11轮早停，第3轮验证macro-F1最佳0.7752。203项累计回归通过，未安装依赖。

已有`src.train`、`src.evaluate`、`src.predict`保留通用检测职责；本次使用独立的`src.train_classification`、`src.evaluate_classification`和`src.predict_classification`。

对本地已裁剪头像预测的示例：

```powershell
.venv/Scripts/python.exe -X utf8 -m src.predict_classification --weights results/classification/goggles_v1_baseline/best.pt --source dataset/goggles/classification_v2_20261022/val/wearing_goggles --name goggles_examples
```

输入应是一张或目录直属的静态头像，每张只有一个头部；输出佩戴状态和两类softmax分数，不输出框。单张1张、目录4张的真实入口检查已完成。不会自动检测头部，分数未经校准。

最终测试29张只评价一次：未佩戴24张中8张误判为佩戴，佩戴5张中3张漏判。完成测试后未改模型或调参。训练运行峰值已分配显存约76.53MiB；完整记录和最佳权重保存在本地`results/`。已有训练/评价目录不覆盖，最终测试记录阻止重复尝试；新实验须使用独立配置、匹配预检和新运行名。

## 文件与数据

| 位置 | 用途 |
| --- | --- |
| [docs/data_processing_plan.md](docs/data_processing_plan.md) | 已执行的清洗、标签、裁剪、来源组、划分和训练前方案 |
| [docs/pretraining_status.json](docs/pretraining_status.json) | 数据准备阶段的历史计数、指纹、无梯度预检和限制摘要 |
| [docs/goggles_training_report.md](docs/goggles_training_report.md) | 真实短跑、基线、学习曲线、两类指标与最终错误记录 |
| [docs/training_status.json](docs/training_status.json) | 当前运行、最佳权重、指标和完整证据哈希摘要 |
| [configs/goggles_classification_v1.yaml](configs/goggles_classification_v1.yaml) | 当前分类数据、初始权重、训练参数与预检绑定 |
| [docs/data_sources/](docs/data_sources/) | 原始来源、逐对象决定、显式裁剪计划和质量记录 |
| [docs/head_crops.md](docs/head_crops.md) | 通用显式头像裁剪与完整性核查接口 |
| [docs/framework_review.md](docs/framework_review.md) | 当前程序职责与训练边界 |

本地正式数据为`dataset/goggles/classification_v2_20261022/`，对应头像`heads_v2_20261022/`和划分`splits_v2_20261022/`。原图、头像、中间预览、完整运行报告、缓存和权重均被Git忽略；只同步代码、配置、来源哈希与文档。v1试裁剪/初始划分保留为中间审计证据，不是当前训练数据。

本轮无需安装依赖；新增依赖按用户要求仅安装到项目虚拟环境。原[训练前快照](docs/pretraining_status.json)保留历史零更新状态，不能作为当前“未训练”的依据。独立记录见[流程修改日志](docs/change_logs/2026-10-22-03-classification-training-workflow.md)与[训练结果日志](docs/change_logs/2026-10-22-04-classification-baseline-results.md)，版本摘要见[CHANGELOG](CHANGELOG.md)。保留独立提交和初版标签`v0.1.0`，按已有授权普通push同步GitHub。
