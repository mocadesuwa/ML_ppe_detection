# 第06轮：三组对照与同一新保留集的准确率比较

- 记录时间：2026-10-22 20:11:34（北京时间，UTC+08:00）。本轮结果提交通过Git历史查找；实际实验代码为第05轮提交`559e2cc81e0157993330acadee731207c57e592c`。
- 原因：按用户授权尝试提高准确率，第05轮已新增110张有效头像、新建v3数据并在训练前限定三组对照。本轮运行该代码、依验证选择、完成一次公平比较及结果文档；没有修改应用代码、数据、标签、划分或阈值。

## 实际训练与选模

使用原项目虚拟环境，未安装依赖。四份配置的匹配CUDA无梯度预检全部通过后，执行`.venv/Scripts/python.exe -X utf8 -u .cache/run_goggles_v2_experiments.py`退出0。该启动器先显式调用`python -m src.train_classification --config configs/goggles_classification_v2_smoke.yaml --start`，两轮26次更新、全部损失有限、骨干/缓冲区冻结、头更新和最佳权重重载一致均断言通过后，才启动control/groups/uniform三份配置的`--start`。

最多并行两个独立训练进程，均从同一ImageNet权重独立初始化，非接着短跑权重训练。MobileNetV3 Small骨干冻结、224、batch16、lr0.0003、weight decay0.0001、种子42、最多30轮、patience8相同；只按预先限定的抽样/类别加权方案对照。

| 配置 | 完成轮数 | 更新 | 最佳轮次 | 最佳验证Accuracy | 最佳验证macro-F1 | 佩戴Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control：逐图shuffle/类别加权 | 11 | 143 | 3 | 91.49% | 0.849359 | 66.67% |
| groups：来源组均衡/类别加权 | 14 | 182 | 6 | 87.23% | 0.713415 | 33.33% |
| uniform：来源组均衡/不加权 | 14 | 182 | 6 | 82.98% | 0.552381 | 11.11% |

三组均为连续8轮验证macro-F1未提高后早停，全部损失有限、骨干参数/BatchNorm缓冲区不变、分类头改变、最佳权重重载logits一致。各运行峰值已分配76.5322MiB；入口内耗时短跑108.65秒，control274.31秒、groups550.88秒、uniform488.21秒，包含输入核查/重载与并行资源竞争，不作为算法训练速度比较。完整日志为`.cache/train-goggles-v2-all.txt`及四份独立训练日志/experiment.json。

来源组均衡使训练佩戴样本的期望占比由44/201≈21.89%变为约10.09%，并把同一工厂120张的期望每轮贡献降至约2.79张。uniform又取消类别加权，验证佩戴Recall很低。该机制与验证现象一致，但不证明全部差异只有一个原因；本轮没有强行采用更复杂却更差的方案。

## 锁定选择、验证与实际图片预测

`.venv/Scripts/python.exe -X utf8 -u .cache/select_evaluate_goggles_v2.py`退出0。先读取三组complete记录，核对协议/配置/权重哈希、更新数和完整验收；严格按最高验证macro-F1、再按该轮Accuracy、仍同分按候选顺序选定control。任何新测试前写入`docs/goggles_improvement_selection.json`，保留训练前`docs/goggles_improvement_protocol.json`不改。

随后实际调用：

```powershell
.venv/Scripts/python.exe -X utf8 -m src.evaluate_classification --config configs/goggles_classification_v2_control.yaml --weights results/classification/goggles_v2_control/best.pt --split val --name goggles_v2_selected_validation
.venv/Scripts/python.exe -X utf8 -m src.predict_classification --weights results/classification/goggles_v2_control/best.pt --source .cache/goggles_v2_examples --name goggles_v2_examples
```

均退出0。验证重评macro-F1、Accuracy与选模记录精确一致，矩阵`[[37,1],[3,6]]`；验证头像两类各2张、共4张预测成功，输出类别和分数、无框。这4张是已有验证成员，不当作新增独立测试。

## 同一新24张保留图片的一次比较

选模锁定后按顺序实际运行，两条命令均退出0：

```powershell
.venv/Scripts/python.exe -X utf8 -m src.evaluate_classification --config configs/goggles_classification_v2_control.yaml --weights results/classification/goggles_v1_baseline/best.pt --split test --final-test --benchmark --name goggles_v1_on_v3_benchmark
.venv/Scripts/python.exe -X utf8 -m src.evaluate_classification --config configs/goggles_classification_v2_control.yaml --weights results/classification/goggles_v2_control/best.pt --split test --final-test --name goggles_v2_selected_test
```

两个模型输入同一24张、相同顺序的新test（未佩戴18、佩戴6），原模型基于其真实旧数据/配置加载，新模型保持选定权重。新test与旧全部191张ID、来源组、PNG字节、像素和原图字节无已知重叠；未选中的groups/uniform没有接触新test。

| 模型 | Accuracy | macro-F1 | 佩戴Precision | 佩戴Recall | 矩阵（真实行/预测列，未佩戴/佩戴） |
| --- | ---: | ---: | ---: | ---: | --- |
| 原v1 | 58.33% | 0.555556 | 33.33% | 66.67% | `[[10,8],[2,4]]` |
| 选定v2 control | 79.17% | 0.705160 | 60.00% | 50.00% | `[[16,2],[3,3]]` |

本次同集Accuracy提高20.83个百分点、macro-F1提高0.149604，错误10张降至5张。未佩戴误报为佩戴从8降至2；佩戴漏判从2增至3，佩戴Recall下降16.67个百分点。因此只称准确率和macro-F1在这批小样本上提高，不称所有指标全面提高；不能与第一轮另一组29张测试的62.07%直接相减。

原v1最佳权重、experiment.json、旧final_test.json、旧29张指标及v2数据清单SHA均与训练前协议一致，未覆盖/重跑旧测试。原模型新增的是`benchmark_<v3指纹>.json`，新模型final_test.json也绑定完整指标SHA；各新test只评价一次，之后没有拟合、改阈值、重选模型或重复推理。

选定权重`results/classification/goggles_v2_control/best.pt`的SHA为`c66432ef67e71efb9341c7aceed7a2291e1f45cf4de313548ad3d9a03c6be8d7`。完整命令/结果在`.cache/goggles-v2-selection-and-comparison.txt`及各阶段日志。

## 文档、回归与核查

新增`docs/goggles_improvement_report.md`、`goggles_improvement_status.json`、测试前选择记录及`docs/assets/goggles_v2_comparison.png`；图表由已有matplotlib生成，实际打开检查曲线、图例、相同色阶的两个混淆矩阵，均与报告一致，仅含聚合数据。README、需求、框架、原方案的历史标记、环境说明、来源索引、AGENTS、CHANGELOG和独立日志索引同步更新。

第05轮应用代码修改后212项累计回归真实通过（22.908秒），包含此前203项。本轮没有改应用代码，因此不重复整套回归。真实新拟合、参数更新、早停、保存/重载、验证、图片预测和公平比较均已实际运行，不能用回归或验证分数代替新test效果。

`python -X utf8 .cache/check_goggles_improvement.py`退出0，`.cache/goggles-improvement-final-checks.json`状态PASS：42份Markdown的UTF-8/围栏、168个本地链接及图像引用、25份日志索引、20份JSON解析通过；272张冻结头像与140张原图/VOC哈希匹配，旧证据不变；协议先于拟合、选择早于两个模型的新测试，协议/配置/代码/权重/完整记录一致；从保存的预测重建指标，确认相同顺序24张、一次性标记、未选模型不测试及佩戴Recall下降均如实记录。该核查没有追加模型前向。

`git diff --check`退出0，仅有Git换行策略提示。`git ls-files -- dataset weights .cache .venv`仍只列出`dataset/raw/.gitkeep`及`weights/.gitkeep`；`git check-ignore`确认新数据清单、最佳权重和完整新测试报告均被隔离。

## 限制与同步

新test只有6个佩戴样本、4个佩戴来源组，单张错误影响佩戴Recall16.67个百分点；未知摄影者/人物关系仍有限制，来源分组不证明完全身份独立。仅一个种子，未完成独立复标、真实实验室域验证或概率校准；不声称统计显著或可部署。后续应优先增加更多独立佩戴场景/姿态和复标，使用新的保留数据，不能继续围绕这24张调参。

本轮未安装依赖。按既有授权普通push代码、配置、元记录、报告与聚合图表，不上传原图、头像、权重、缓存和完整逐图输出；保留v0.1.0及旧提交，不强推。同步结果以本地回执为准，本日志不预填尚未执行的push结果。
