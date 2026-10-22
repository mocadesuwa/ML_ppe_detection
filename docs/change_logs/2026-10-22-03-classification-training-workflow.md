# 第03轮：修正损失统计并补齐分类训练、评价和预测

- 记录时间：2026-10-22 18:24（北京时间，UTC+08:00）。提交编号通过本日志的Git历史查找。
- 修改原因：用户在整体流程分析后确认“按这个模式推进”，授权先修正和补齐入口，2轮短跑验证通过后执行既定基线，最后评价冻结测试集。此授权更新前一阶段不训练的边界，原始数据和冻结划分保持不变。

## 涉及文件与变化

- `src/train_classification.py`：修正带类别权重loss的整轮累计分母；记录未加权验证loss、两类指标、运行模式、代码/环境、更新数和显存/耗时。核对骨干参数及缓冲区不变、分类头改变，最佳检查点重载logits一致才算完成。
- `src/classification_model.py`：明确分类检查点的任务、类别顺序、224填充预处理和数据指纹；完成训练记录绑定最佳权重哈希；输出显式混淆矩阵轴、两类Precision/Recall/F1与zero_division=0。
- `src/evaluate_classification.py`：独立分类验证/最终测试，记录逐图概率和错误；最终测试要求完成的baseline、显式开关、当前配置/数据/权重一致，并在模型前向前独占创建一次性测试记录。
- `src/predict_classification.py`：对已裁剪单张/目录静态头像逐图输出JSONL，不输出框；检查输入、权重、训练记录前后不变，拒绝覆盖。共用原静态图片检查函数，没有调用旧检测模型。
- 两份分类配置：基线保持原超参数，增加baseline模式和新预检路径；独立smoke配置仅改变模式、两轮上限、运行名和匹配预检路径，二者从原ImageNet权重分别初始化。
- `tests/test_classification_workflow.py`新增14项：批次分组不影响整轮加权loss、全负类高Accuracy反例、指标/概率、检查点契约、最终测试开关、smoke拒绝、变更/未验证运行拒绝和重复测试拒绝。
- `AGENTS.md`、需求、框架、README与新增`docs/classification_training_workflow.md`同步当前授权、执行规则和未定义效果门槛；不声称真实实验室验收通过。

## 实际验证与前序回归

- `python -X utf8 tests/run_regression.py`退出0：203项，21.922秒，全部通过。完整输出`.cache/regression-2026-10-22-classification-workflow-01.txt`。包含前序189项，未跳过既有坐标、解压、分组、完整性、入口、预测和头像准备回归。
- 三个分类CLI的真实`--help`子进程均退出0；`compileall`对src/tests通过。
- 两份配置的真实CUDA无梯度预检在已有`.venv`均退出0，分别保存`results/preflight_goggles_v1_smoke/report.json`和`results/preflight_goggles_v2/report.json`：`ready_before_training`、更新0、两类16张前向通过、参数及缓冲区不变，峰值已分配54.7158MiB。旧预检保留。
- 文档检查退出0：100个本地链接存在、22份独立日志均入索引、UTF-8/代码围栏及docs内JSON解析通过，证据`.cache/classification-workflow-doc-checks.json`。`git diff --check`退出0，仅提示既有Git换行策略。
- 本日志记录时尚未启动梯度训练、正式评价或最终测试；后续真实结果另建日志，不把203项回归当作模型效果。

## 限制与下一步

训练133张只有12个显式来源组，佩戴正例覆盖3组；工厂来源占120/133。val/test各5个正例，全部预测未佩戴也有82.76%Accuracy。独立复标、实际实验室域验证和课程效果门槛未完成，不在本轮擅自扩充或改划分。

回归通过后先执行两轮短跑；有限损失、18次更新、骨干冻结、头更新和最佳权重重载一致全部成立后才执行最多30轮基线。先验证和演示头像预测，最终一次冻结测试，不根据测试指标重新调参。本轮不安装依赖；完整运行证据、数据和权重不上传。
