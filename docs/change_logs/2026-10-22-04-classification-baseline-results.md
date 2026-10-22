# 第04轮：两轮短跑、基线训练与一次冻结测试

- 记录时间：2026-10-22 18:40（北京时间，UTC+08:00）。本次结果文档的关联提交通过Git历史查找；实际训练代码提交`8b3175786120cb07b0f06bec0a61131a09682050`。
- 原因：按用户确认的整体流程，先验证真实训练过程，通过后执行既定基线，再评价冻结测试。第03轮已修正日志和补齐分类入口，本轮不修改代码、原始数据、标签、裁剪、来源组或划分。

## 实际执行与结果

使用已有`.venv`、torch2.11.0+cu128、torchvision0.26.0+cu128与RTX3060 Laptop GPU；没有安装/升级依赖，没有改系统Python。训练参数仍为冻结骨干MobileNetV3 Small、224、batch16、AdamW lr0.0003、weight decay0.0001、种子42，类别权重只按训练97/36计算。

1. 两份配置匹配预检核查通过。两份配置SHA分别为短跑`ea4e598a61e22e145825cba31eed2e7d313668b6b5ed57155dc93dd7d7b49e13`、基线`64bb40122ba0bfe5f0df1ec54ebe8fef0bf94cb5c9112222f8c8b397d43077b2`；数据指纹保持`b9176dd923d011a63f7373e126d0997f9959c2e324debc20117140013250e7af`。
2. `.venv/Scripts/python.exe -X utf8 -u -m src.train_classification --config configs/goggles_classification_smoke_v1.yaml --start`退出0：2轮18次更新，全部损失有限，骨干参数/BatchNorm缓冲区哈希不变、分类头改变，最佳权重重载logits一致；最佳第2轮验证macro-F1为0.741071。训练记录计时46.8729秒，峰值已分配76.5322MiB；完整输出`.cache/train-goggles-smoke-20261022.txt`。
3. 启动器读取短跑实际记录，对18次更新、2轮、有限损失、三项冻结/更新/重载条件全部断言通过后，才执行`.venv/Scripts/python.exe -X utf8 -u -m src.train_classification --config configs/goggles_classification_v1.yaml --start`。退出0：基线独立从ImageNet权重开始，完成11轮99次更新；第3轮验证macro-F1最高0.775194，连续8轮未提高早停。冻结、头更新和重载再次通过；训练记录计时84.8246秒，峰值已分配76.5322MiB；输出`.cache/train-goggles-baseline-20261022.txt`。
4. `.venv/Scripts/python.exe -X utf8 -u -m src.evaluate_classification --config configs/goggles_classification_v1.yaml --weights results/classification/goggles_v1_baseline/best.pt --split val --name goggles_v1_validation`退出0。最佳权重验证重评macro-F1与选模记录精确相同，Accuracy82.76%，混淆矩阵`[[19,5],[0,5]]`；输出`.cache/evaluate-goggles-validation-20261022.txt`。
5. 实际调用`src.predict_classification --weights results/classification/goggles_v1_baseline/best.pt --source <验证头像> --name goggles_v1_single`以及目录模式`--source .cache/goggles_prediction_examples --name goggles_v1_directory`，均使用项目虚拟环境，退出0。单张1张，目录4张（两类各2张），JSONL数量、索引顺序、两类分数和无框契约通过；原图字节复制到忽略的演示目录，不加入数据集。输出`.cache/predict-goggles-20261022.txt`。
6. 启动器确认基线complete、验证重评一致和两种预测成功，才调用`.venv/Scripts/python.exe -X utf8 -u -m src.evaluate_classification --config configs/goggles_classification_v1.yaml --weights results/classification/goggles_v1_baseline/best.pt --split test --final-test --name goggles_v1_final_test`。退出0：29张一次评价完成，Accuracy62.06897%、macro-F1 0.505426，测试矩阵`[[16,8],[3,2]]`。未佩戴Precision84.21%/Recall66.67%/F1 74.42%，佩戴Precision20%/Recall40%/F1 26.67%。输出`.cache/evaluate-goggles-final-test-20261022.txt`。

最终测试前独占创建`results/classification/goggles_v1_baseline/final_test.json`，完成后绑定指标报告SHA。没有重跑测试，没有根据测试改模型、轮次、阈值或数据。基线最佳权重SHA为`ab649ad5020a80e9f0edcd00a9fe75e808b4ed3e3df2d29bc6a4d0e06f4737eb`。

## 文档、图表和错误分析

新增`docs/training_status.json`绑定训练代码版本、运行环境、更新数、两类指标、最佳权重和7份完整证据SHA；新增`docs/goggles_training_report.md`及聚合图`docs/assets/goggles_v1_training_results.png`，同步README、实施方案、需求、框架、数据处理方案、环境说明、训练流程和CHANGELOG。原`docs/pretraining_status.json`与前序日志作为历史快照保留，没有把旧零更新结果改写为当前状态。

图表使用已有matplotlib3.11.2生成，并实际打开核对坐标、图例、最佳轮次和测试矩阵；只含聚合数据，不含人物照片。训练loss按类别权重平均，验证loss不加权，图注注明两者口径不同。

测试共11个错误ID全部列入报告。Codex事后只读抽看4张：两张低头无防护镜误报、两张可见防护镜的低头/侧向头像漏判；只是4例观察，不认定因果，不改标签，也没有为查看错误而追加模型推理。当前测试Accuracy低于全负预测的82.76%参考值，佩戴Precision/Recall较弱，明确记录泛化不足，不把完成流程称为效果验收通过。

## 验证与前序回顾

第03轮代码修改后真实累计回归203项全部通过（21.922秒），保留前序189项。本轮仅执行已验证代码、生成结果和修改文档，没有代码变化，因此不无故重复整套回归。模型真实拟合、更新、早停、保存/重载、验证、单张/目录预测和最终测试已按上述命令实际运行；这些证据与单元回归分开记录。

`python -X utf8 .cache/check_classification_results.py`退出0，`.cache/goggles-training-final-checks.json`状态PASS：38份Markdown的UTF-8/围栏、127个本地链接及图片引用、23份日志索引、12份JSON解析通过；191张冻结图像SHA全部匹配；训练代码/配置/最佳权重与7份完整证据摘要一致；从保存的29张测试预测重建指标与摘要完全一致，验证结果与最佳轮次相同，单张/目录预测的实际来源排序、数量和哈希通过。此检查没有追加模型前向。

`git diff --check`退出0，仅有Git换行策略提示。`git ls-files -- dataset weights .cache .venv`只列出`dataset/raw/.gitkeep`和`weights/.gitkeep`；`git check-ignore`确认最佳训练权重、完整最终测试报告、冻结数据清单和初始权重均仍被忽略。图表已实际打开核对，没有重复进行测试推理。

提交前`git fetch origin`退出0，`origin/main...HEAD`为0落后/1领先，对应第03轮代码提交；初版`v0.1.0`仍为`f6e9a6859893027e3fababe3697838723aa3a875`。本轮按既有授权普通push同步代码与结果摘要，不强推、不上传原图、头像、完整逐图报告、缓存或权重；同步回执保存在本地缓存，日志不预填尚未执行的push结果。

## 限制和后续

本次已完成确认的小样本基线流程；当前效果不足以支持稳定使用。训练133张只有12个显式来源组，佩戴样本覆盖3组，120张来自工厂来源；val/test各仅5个佩戴样本，未知人物/摄影者关系仍不完全掌握。没有独立第二标注者复核、真实实验室场景验证、分数校准或课程效果门槛。

下一阶段重点是独立来源/实际实验室样本与复标，扩充姿态和防护镜类型覆盖，建立新数据版本与新的保留评价集；保留本次冻结测试和结果，不能围绕这29张反复调参。本轮未进行这些新增数据工作。
