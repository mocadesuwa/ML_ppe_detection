# 实验室防护用品佩戴检测

机器学习课程实践项目，首版子任务为人脸口罩状态检测：mask、no_mask、mask_incorrect。使用 YOLO11n 预训练检测模型，后续微调到三类。

**已完成到首次训练前（2026-10-08）**：派生数据、固定划分、实际标注抽查、CUDA 环境、初始权重和无梯度 GPU 前向检查均完成。短跑与正式训练未启动，暂无本任务模型指标。

初版标签 **v0.1.0** 保留。每轮独立[修改日志](docs/change_logs/README.md)与代码一起提交，版本摘要见 [CHANGELOG](CHANGELOG.md)。本轮按用户要求先完成阶段检查，结尾统一累计回归：系统 Python 和项目 `.venv` 均 141 项通过。

## 当前结果

- 原始 853 对图片/XML 和 ZIP 保留；新副本 794 张、3,510 个目标，逐项记录排除、去重及边缘修正。
- 当前 `dataset/prepared_v2`：train/val/test **510/206/78** 张，579 个独立组。全量程序检查通过，最终各集合实际查看 30 张标注图，质量记录绑定指纹。
- 三类目标数分别为 train **1983/439/39**、val **565/121/11**、test **286/61/5**；少数类样本很少。
- `.venv`：Python 3.14、torch 2.11.0+cu128、torchvision 0.26.0+cu128、Ultralytics 8.4.174；RTX 3060 Laptop 6 GiB、CUDA 可用，依赖检查通过。41 包版本见 [requirements-lock.txt](requirements-lock.txt)。
- 官方初始权重发布方 SHA-256 匹配；真实训练集加载、8×3×640×640 无梯度 GPU 前向和 CUDA NMS 通过。此检查不包含反向传播、优化器、训练显存或效果评价。

证据：[训练前准备报告](docs/pretraining_ready.md)、[固定划分核查](docs/data_reviews/2026-10-08-prepared-v2.md)、[派生决策](docs/data_reviews/2026-10-08-derived-v1.md)、[环境与复现](docs/environment_check.md)。

## 运行与停止点

在项目根目录使用项目解释器；中文路径已通过真实数据加载检查：

```powershell
# 只查看计划，不加载模型或创建训练目录。
.\.venv\Scripts\python.exe -m src.train --dry-run
.\.venv\Scripts\python.exe -m src.train --smoke --dry-run

# 重做无梯度预检须选择新证据目录。
.\.venv\Scripts\python.exe scripts/preflight_training.py --output results/preflight_new

# 累计回归仅使用缓存中的临时合成样本。
.\.venv\Scripts\python.exe tests/run_regression.py
```

下一步是首次 GPU 短跑，须用户明确启动后执行。随后才做正式 baseline、验证分析和最终测试。计划仍为 imgsz=640、batch=8、最多50轮、workers=0；实际训练显存不足时降低 batch 并记录参数。前向检查显存不能当作训练用量。

## 文件与文档

| 位置 | 职责 |
| --- | --- |
| configs/ | 当前数据版本、三类、派生决策、画面组及训练计划 |
| src/ | 下载、转换、核查、环境、训练、评价和预测入口 |
| scripts/ | 原始核查、可追溯派生、画面候选及无训练预检 |
| tests/ | 坐标、分组、解压、指纹、配置及模型入口回归 |
| dataset/raw/ | 原始图片/XML、ZIP与来源记录 |
| dataset/derived/source_v1/ | 派生图片/XML及操作台账 |
| dataset/prepared_v2/ | 当前冻结图片、YOLO标签、划分、清单及质量记录 |
| weights/ | 初始权重；任务训练权重未生成 |
| results/ | 核查及预检证据；无 smoke/baseline 训练目录 |

真实数据、缓存、虚拟环境和权重被 Git 忽略；仓库保存代码、决策、摘要和来源哈希。复现已存在的派生版本须复制配置并选新目录，不能覆盖冻结清单，也不能未经实际查看就复制质量记录文字。

继续阅读：[框架说明](docs/framework_review.md)、[Baseline 方案](docs/baseline_v0.1.md)、[数据选择](docs/data_selection.md)、[实验记录模板](docs/experiment_record.md)。

当前是通用场景口罩子任务，尚无实验室、手套或护目镜实测。少数样本、来源不完整和验证画面分布变化已记录，后续报告需解释这些限制。
