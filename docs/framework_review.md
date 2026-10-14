# 护目镜项目代码框架

更新日期：2026-10-14。当前输入方向为静态头部裁剪头像，位置框要求待确认；最终任务类型以[需求清单](goggles_requirements.md)为准。当前没有护目镜实测模型或指标。

## 代码职责

| 文件 | 职责 |
| --- | --- |
| `src/common.py` | 路径、配置、哈希、数据指纹、质量记录及实验目录 |
| `src/import_data.py` | 显式本地ZIP导入、安全解压及来源记录，不联网 |
| `src/source_review.py` | 显式类别的VOC源数据只读核查：尺寸、对象、两种坐标约定、重复冲突与来源指纹 |
| `src/prepare_dataset.py` | VOC到YOLO转换、去重、来源分组与固定划分 |
| `src/check_dataset.py` | 检测标签、清单和划分校验，生成标注抽查拼图 |
| `src/review_dataset.py` | 记录实际人工核查并绑定数据指纹 |
| `src/environment.py` | 查询现有Python、依赖和GPU环境 |
| `src/train.py` | 显式训练计划，`--dry-run`不读取真实数据内容或加载模型 |
| `src/evaluate.py` | 显式数据/权重、类别一致性、训练来源及最终测试约束 |
| `src/predict.py` | 静态图片预检、逐张预测及JSONL/成功失败记录 |
| `scripts/find_related_images.py` | 相似图片候选，不自动判断人物或来源组 |
| `scripts/preflight_training.py` | 显式检测配置的无梯度预检，不能代替训练或任务评价 |

数据框转换、训练、评价和预测目前仍包含检测模型约束。若最终采用分类，需同步调整任务检查、数据结构与指标；不默认扩展为同时支持多种模式的框架。

## 源数据核查

`python -m src.source_review`必须提供`--raw`、`--classes`和`--output`。类别名称与源XML精确对应；未知对象记录错误，不能静默跳过。两种坐标约定分别检查，不自动选定哪种为真值。

输出包含类别表、全部对象、重复冲突、来源哈希与前后不变检查；拒绝覆盖已有输出或写入原始数据内。报告明确`visual_review_completed=false`，程序核查不等同于人工看过图片。本工具不修改、修补或裁剪原始数据。

## 图片结果格式

- 支持JPG/JPEG、PNG、BMP和静态WebP。空文件、无法校验的图片、动画及单文件视频在模型加载前报错。
- 目录只筛选直接包含的支持图片，逐张交给后端；没有图片时报错，不一次加载整个数据集。
- `predictions.jsonl`每张包含`source`、从0开始的`image_index`、`image_size`和`detections`；检测项包含类别、置信度与像素框。空检测不能直接表示确定未佩戴。
- `run.json`使用`format_version=2`、`image_count`，绑定输入/权重哈希与实际参数，记录`running/complete/failed`。
- 保留重复/遗漏、框/类别/置信度及内容变化检查；结果先写`.part`，通过完整检查后才发布，失败或中断保留证据。

图片入口只能核查格式，不能认定输入已经正确裁成头部头像。最终护目镜输出应随任务类型一起确定。

## 验证与下一步

回归使用临时合成样本和假模型，中性类别名仅为测试数据。按既定顺序检查坐标、分组、解压、清单/指纹、标签、配置、评价、预测和来源核查；代码变化后运行`python tests/run_regression.py`。

当前验证记录见[第03轮日志](change_logs/2026-10-14-03-goggles-only.md)。下一步明确输出与头像/标签规则，再推进护目镜数据、配置和模型；尚未启动真实数据处理、依赖安装或训练。第一版完成前不上传GitHub。
