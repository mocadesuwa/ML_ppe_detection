# 实验室护目镜佩戴识别

机器学习课程实践项目：对**静态头部裁剪头像进行二分类**，只输出佩戴状态，不输出位置框。用户已确认：防护镜正确覆盖眼部为佩戴；普通眼镜、太阳镜、额头佩戴为未佩戴；无法判断的排除。

当前已完成**数据扩充、短跑、三组对照和同一新保留集的比较**：24张图片上，Accuracy从旧模型58.33%升至79.17%，macro-F1从0.5556升至0.7052；佩戴召回率从66.67%降至50%，存在取舍。详见[第二轮改进报告](docs/goggles_improvement_report.md)与[当前状态](docs/goggles_improvement_status.json)。样本仍小，未证明实际实验室部署效果。

第二轮新增110张已复核头像，累计301张有效头像；封存旧29张测试，建立272张v3实验数据。按验证集选择了扩充数据后沿用原抽样/类别加权的control，来源组均衡的两个替代方案没有额外收益；[选择规则](docs/goggles_improvement_protocol.json)在训练前锁定，测试前已保存[选定记录](docs/goggles_improvement_selection.json)。

第一轮180张原图产生191张头像和133/29/29划分，仍保留为历史版本，见[原数据方案](docs/data_processing_plan.md)和[第一轮报告](docs/goggles_training_report.md)。当前v3划分如下：

| 集合 | 未佩戴 | 佩戴 | 合计 |
| --- | ---: | ---: | ---: |
| train | 157 | 44 | 201 |
| val | 38 | 9 | 47 |
| 新test | 18 | 6 | 24 |

所有工厂连续帧保持同组进入训练集；训练集也包含SH17的两类样本。训练来源组增至72，佩戴样本覆盖8组；新test有20组，6个佩戴样本来自4组。已知来源组无交叉，新test与旧全部191张在ID、分组、原图/头像字节及像素上无已知重叠；未知人物/摄影者关系仍有限制。

## 当前入口

主流程：原图与原标注 → 逐项复核及头部框 → 头像裁剪与像素核查 → 按来源整组划分 → 分类数据导出与预检 → 训练、评价与头像预测。

```powershell
python tests/run_regression.py
python -m src.classification_data check --data dataset/goggles/classification_v3_20261022
python -m src.train_classification --config configs/goggles_classification_v2_control.yaml --dry-run
```

分类训练入口默认只核查并打印计划，真实训练必须显式提供`--start`，且要求匹配预检。第二轮短跑2轮26次更新通过后执行三组对照，均通过骨干冻结、头更新和最佳权重重载。选定control第11轮早停，第3轮验证macro-F1最佳0.8494、Accuracy91.49%；212项累计回归通过，未安装依赖。

护目镜入口是`src.train_classification`、`src.evaluate_classification`和`src.predict_classification`。通用检测工具及共享校验仍保留，具体职责见[框架说明](docs/framework_review.md)。

对本地已裁剪头像预测的示例：

```powershell
.venv/Scripts/python.exe -X utf8 -m src.predict_classification --weights results/classification/goggles_v2_control/best.pt --source dataset/goggles/classification_v3_20261022/val/wearing_goggles --name goggles_v2_new_examples
```

输入应是一张或目录直属的静态头像，每张只有一个头部；输出佩戴状态和两类softmax分数，不输出框。第一轮单张/目录和第二轮选定模型4张验证头像均实际核验通过；不会自动检测头部，分数未经校准。

新保留集只对原模型和选定模型各评价一次：未佩戴误判佩戴由8张降至2张，佩戴漏判由2张增至3张。测试后没有调整模型或阈值；旧29张测试没有重跑。训练运行峰值已分配显存约76.53MiB，完整证据和权重仅存本地`results/`。已有目录不覆盖；新实验须用新配置、匹配预检和独立保留评价，不能借benchmark重置旧测试。

## 文件与数据

| 位置 | 用途 |
| --- | --- |
| [dataset/README.md](dataset/README.md) | 本地数据目录、当前版本与保留原因 |
| [configs/README.md](configs/README.md) | 当前、对照及历史配置的用途 |
| [docs/data_processing_plan.md](docs/data_processing_plan.md) | 已执行的清洗、标签、裁剪、来源组、划分和训练前方案 |
| [docs/pretraining_status.json](docs/pretraining_status.json) | 数据准备阶段的历史计数、指纹、无梯度预检和限制摘要 |
| [docs/goggles_training_report.md](docs/goggles_training_report.md) | 第一轮基线及旧29张测试结果，保留历史 |
| [docs/goggles_improvement_report.md](docs/goggles_improvement_report.md) | 第二轮三组对照、选模、相同新测试的比较与取舍 |
| [docs/goggles_improvement_status.json](docs/goggles_improvement_status.json) | 当前运行、最佳权重、指标与完整证据SHA摘要 |
| [configs/goggles_classification_v2_control.yaml](configs/goggles_classification_v2_control.yaml) | 当前选定方案的数据、初始权重、参数与匹配预检 |
| [docs/data_sources/](docs/data_sources/) | 原始来源、逐对象决定、显式裁剪计划和质量记录 |
| [docs/head_crops.md](docs/head_crops.md) | 通用显式头像裁剪与完整性核查接口 |
| [docs/framework_review.md](docs/framework_review.md) | 当前程序职责与训练边界 |

本地当前数据为`dataset/goggles/classification_v3_20261022/`，对应头像`heads_v3_20261022/`与划分`splits_v3_20261022/`。原图、头像、完整逐图报告、缓存和权重被Git忽略，只同步代码、配置、来源/审核哈希及聚合结果。旧v1/v2头像和划分、第一轮模型与测试保留历史。

旧口罩专用原始数据、压缩包、派生数据和检查输出已从本地清理。当前护目镜使用的PPEs、SH17及其原标注保留；混合PPE来源中的其他类别不作为本项目输出。历史日志仅供审计，详见[本轮清理记录](docs/change_logs/2026-10-22-07-remove-mask-data.md)。

本轮无需安装依赖；新增依赖仅装项目虚拟环境。原[训练前快照](docs/pretraining_status.json)及[第一轮运行](docs/training_status.json)保留历史。训练与改进证据见[扩充与代码](docs/change_logs/2026-10-22-05-expansion-and-controlled-experiments.md)及[改进结果](docs/change_logs/2026-10-22-06-improvement-results.md)，版本摘要见[CHANGELOG](CHANGELOG.md)。保留独立提交和初版标签`v0.1.0`，按既有授权普通push。
