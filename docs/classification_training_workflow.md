# 头像分类训练与评价流程

更新日期：2026-10-22。用户已确认按“修正日志和补齐入口→短跑→既定基线→冻结测试”的顺序推进。本页记录执行契约；现已完成，真实结果见[训练报告](goggles_training_report.md)和[运行摘要](training_status.json)。以下命令是本次执行记录，既有运行不会覆盖，最终测试不能重复。

本页为第一轮历史执行契约；当前第二轮数据扩充、预先限定的三组对照与新保留集约束另见[改进方案](goggles_improvement_plan.md)，真实结果见[改进报告](goggles_improvement_report.md)。

## 数据和选择规则

- 沿用191张头像和133/29/29冻结划分，指纹见[数据处理方案](data_processing_plan.md)，不扩充、重标或拆分来源组。
- train有12个来源组，佩戴正例覆盖3组；120/133张来自工厂来源组。val/test各29张，其中佩戴5张。全预测未佩戴的Accuracy为24/29≈82.76%，不能单看Accuracy验收。
- 保持MobileNetV3 Small冻结骨干、224等比例填充、batch16、AdamW学习率0.0003、种子42。短跑2轮；基线最多30轮、patience8。两个运行均从同一ImageNet权重独立初始化，基线不接着短跑权重训练。
- 只按验证macro-F1严格提高保存最佳；同分保留更早轮次。测试不参与选轮次、超参数、阈值或类别权重。
- 当前未设课程效果合格门槛。先如实报告两类Precision/Recall/F1、macro-F1、Accuracy、混淆矩阵和错误样本；独立复标及实际实验室域验证尚未完成。

## 训练前修正

带类别权重的CrossEntropyLoss默认按样本权重之和归一化。整轮日志按`sum(batch_loss × batch_weight_sum) / sum(batch_weight_sum)`累计，修正原先按人数加权的日志偏差；不改变每批反向传播的损失定义。参考[PyTorch 2.11官方说明](https://docs.pytorch.org/docs/2.11/generated/torch.nn.CrossEntropyLoss.html)。验证loss为不加类别权重的平均交叉熵。

新增分类专用的[评价入口](../src/evaluate_classification.py)和[图片预测入口](../src/predict_classification.py)，共用[检查点与指标契约](../src/classification_model.py)。已有通用检测入口保留原职责。

## 执行与可检查条件

在项目虚拟环境执行。两个配置各有匹配的无梯度预检报告，修改配置后须生成新的匹配报告，不能覆盖旧证据。

```powershell
.venv/Scripts/python.exe -X utf8 -m src.train_classification --config configs/goggles_classification_smoke_v1.yaml --start
.venv/Scripts/python.exe -X utf8 -m src.train_classification --config configs/goggles_classification_v1.yaml --start
```

短跑须满足：两轮完成、损失有限、133张每轮按batch16产生9次更新，共18次；骨干参数及BatchNorm缓冲区哈希不变；分类头改变；最佳检查点重载后的验证探针logits与保存时一致。记录环境、代码哈希、梯度更新数、各轮损失和验证指标、训练运行峰值显存与耗时。短跑合格只证明流程可运行。

基线通过完整训练记录和最佳检查点重载检查后，先重新评价验证集并演示头像预测；评价集仍不增强，统一argmax，不使用测试集调阈值。最终只运行一次：

```powershell
.venv/Scripts/python.exe -X utf8 -m src.evaluate_classification --config configs/goggles_classification_v1.yaml --weights results/classification/goggles_v1_baseline/best.pt --split test --final-test --name goggles_v1_final_test
```

最终测试必须来自完成的baseline，拒绝smoke、未完成运行、检查点/数据/配置变化；在模型前向前独占创建该运行的`final_test.json`。已有尝试（包括失败）时读取原证据，不静默重跑。训练记录不被评价程序改写，测试结果另存。

预测入口接收单张或目录直属的静态头部头像，逐张输出类别、两类softmax分数和输入哈希，无位置框；不自动检测或裁剪头部。softmax分数未经校准，不等于实际正确率。输出与完整记录使用新目录，拒绝覆盖，输入/权重变化则不发布完整结果。

## 证据与限制

完整运行证据和权重只存本地`results/`；同步可复核的摘要、配置、代码及独立中文日志，不上传原图、头像或模型文件。上一阶段[训练前状态](pretraining_status.json)保留为历史快照，里面的零更新和旧配置哈希只描述当时的无梯度预检。

公开工业/素材样本与真实实验室存在场景差异，样本之间可能还有未掌握的关系；来源组无交叉不等于人物身份绝对独立。每个评价集5个正例，正类错1张就使召回率变化20个百分点。获得指标后仍须保留这些限制，不把小样本基线称为部署完成。
