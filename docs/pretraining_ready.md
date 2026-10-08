# 首次训练前准备完成记录

北京时间 2026-10-08。已停在首次训练前；没有执行短跑、正式训练、反向传播或优化器更新。

## 已完成与证据

| 工作 | 实际结果 | 证据 |
| --- | --- | --- |
| 原件保留与派生 | 原件 853 对及 ZIP 未变，新副本 794 对、3,510 个目标 | [派生决策](data_reviews/2026-10-08-derived-v1.md) |
| 画面分组与固定划分 | 579 组；train/val/test 510/206/78，人工组跨集合 0，确定性重算一致 | [划分报告](data_reviews/2026-10-08-prepared-v2.md) |
| 标签与实际查看 | 全量检查无错误；最终各集合 30 张实际查看，质量记录绑定指纹 | [数据摘要](data_reviews/prepared_v2_summary.json) |
| 项目环境 | CUDA 可用，矩阵运算通过，pip check 无破损要求；41 包锁定 | [环境说明](environment_check.md) |
| 初始权重 | YOLO11n 官方 asset 大小和发布方 SHA-256 均匹配 | [来源记录](initial_weights.json) |
| 不训练的真实预检 | 训练集加载 510 张有效，batch=8/640 无梯度前向和 CUDA NMS 通过，内容未变 | [原始预检结果](pretraining_check.json) |
| 结尾累计回归 | 系统 Python 141 项/6.709 秒；项目 .venv 141 项/6.864 秒，均通过 | [14 日志](change_logs/2026-10-08-14-pretraining-ready.md) |

最终数据指纹：`fa0dad948f835b862633b2451915085b24f65f14a4f1cb6ab657dac10e4f5a7f`。`results/preflight_01/report.json` 的 `training_started=false`、`gradient_updates=0`；`results/smoke_01` 与 `results/baseline_01` 均不存在。最初回归发现一项旧默认目录断言，更新为实际冻结版本后才获得上述通过结果。

## 后续真正的第一步

用户明确启动训练后，先做 GPU 短跑，命令预先准备如下；**本轮没有执行**：

```powershell
.\.venv\Scripts\python.exe -m src.train --smoke
```

短跑成功后再运行正式 baseline，保持固定数据版本，记录实际参数、环境、best/last 权重、验证结果和耗时。不要把预训练 80 类权重直接当作已训练的三类 PPE 模型；也不要先跑最终测试指标来调整方案。

## 当前限制

三类少数样本在 val/test 仅 11/5 个目标；181 张摆拍集合整体在 val，图片比例和画面分布因此变化。来源元数据不完整，程序与人工组不能证明全部共同来源均已找出。其余类未全量人工复标，新闻/广告照片、密集小目标和场景差异仍存在。

无梯度前向验证了当前实际运行路径，不替代训练显存、反向传播、优化器、数据增强、权重保存或任务效果验证。首版仍为口罩子任务，实验室场景及多用品任务留到有数据和课程要求后推进。
