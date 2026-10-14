# 实验室护目镜佩戴识别

机器学习课程实践项目，输入使用**静态头部裁剪头像**，识别护目镜佩戴情况。是否需要位置框仍待确认，最终分类/目标检测路线、类别和裁剪细则尚未确定，详见[需求清单](docs/goggles_requirements.md)。

**当前阶段（2026-10-14）：已完成旧任务清理和静态图片入口整理。** 尚无护目镜数据版本、训练模型或指标。原有检测后端暂时保留供适配，不代表已选定护目镜模型。

## 已调整的内容

- 口罩配置、来源决策与两份专用脚本移入 [legacy/mask](legacy/mask/README.md)，不再作为当前默认入口。
- 删除固定下载口罩数据的网络逻辑；`src/import_data.py` 仅导入显式提供的本地 ZIP，保留来源、哈希和安全解压检查。
- 准备、核查、训练和评价入口须显式提供配置；预测和评价须显式提供权重。
- 图片预测只处理 JPG/JPEG、PNG、BMP 和静态 WebP，拒绝视频、动画和无法读取的图片。目录不递归，逐张处理筛选后的图片。
- 保留去重、来源分组、标签校验、数据指纹、质量核查及失败记录等通用能力；没有把旧口罩标签改成护目镜标签。

## 目录

| 位置 | 用途 |
| --- | --- |
| [src/](src/) | 通用数据导入、VOC/YOLO数据处理、环境和保留的检测后端 |
| [scripts/](scripts/) | 来源相似图片候选与检测训练前预检，均须显式指定输入 |
| [configs/](configs/README.md) | 后续护目镜配置位置，当前不提供未经确认的默认训练方案 |
| [tests/](tests/) | 合成数据与假模型累计回归，包括归档能力的保护检查 |
| [legacy/mask/](legacy/mask/README.md) | 口罩旧任务配置、决策与专用脚本，仅供历史复现 |
| [docs/framework_review.md](docs/framework_review.md) | 文件职责、图片结果格式和当前边界 |
| [docs/change_logs/](docs/change_logs/README.md) | 每轮独立中文日志；版本摘要见 [CHANGELOG](CHANGELOG.md) |

## 检查与使用入口

在项目根目录使用已有依赖的系统 Python；未安装新依赖。以下命令不训练、不修改真实数据：

```powershell
python tests/run_regression.py
python -m src.import_data --help
python -m src.predict --help
python -m src.train --help
```

本轮完整回归在正常本机权限下 **146项通过**，记录和首次沙箱运行未完成的情况见[第02轮日志](docs/change_logs/2026-10-14-02-static-image-cleanup.md)。测试只使用临时合成样本与假模型，不能说明护目镜识别效果。

当前不处理真实数据、不安装依赖、不启动短跑或正式训练。后续先明确输出、头像/裁剪与标签规则，再推进数据准备和模型适配。护目镜第一版验收前不上传 GitHub，本地独立提交及日志继续保留；初版标签 **v0.1.0** 保留。

## 历史证据

旧口罩数据 `dataset/prepared_v2` 及原始文件保留，未修改、未重新解释标签。旧任务未启动训练；2026-10-08的GPU、数据核查和141项测试是历史记录，本轮未重新检查真实数据或模型环境。

参考：[旧训练前准备](docs/pretraining_ready.md)、[旧数据核查](docs/data_reviews/2026-10-08-prepared-v2.md)、[旧Baseline方案](docs/baseline_v0.1.md)。旧文档中的路径和命令描述当时版本，当前复现入口见归档说明。
