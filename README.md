# 实验室护目镜佩戴识别

机器学习课程实践项目：对**静态头部裁剪头像**识别护目镜佩戴情况。是否需要位置框仍待确认，最终分类/目标检测路线、类别及裁剪细则见[需求清单](docs/goggles_requirements.md)。

当前是护目镜项目的基础代码阶段，尚无已核查的护目镜数据版本、训练模型或指标。图片入口目前保留检测后端，最终模型尚未确定。

## 当前目录

| 位置 | 用途 |
| --- | --- |
| [src/](src/) | 本地数据导入、显式类别源数据核查、数据准备、训练/评价与图片预测 |
| [scripts/](scripts/) | 相似图片候选与无梯度预检，须显式指定输入 |
| [configs/](configs/README.md) | 护目镜配置位置，任务契约明确后建立 |
| [tests/](tests/) | 使用中性标签、临时合成数据和假模型的通用回归 |
| [docs/framework_review.md](docs/framework_review.md) | 当前代码职责和图片结果格式 |
| [docs/goggles_baseline.md](docs/goggles_baseline.md) | 护目镜第一版实施方案与待确认项 |
| [docs/data_selection.md](docs/data_selection.md) | 护目镜数据选择、头像和划分规则 |
| [docs/experiment_record.md](docs/experiment_record.md) | 护目镜实验记录模板，未填写实测结果 |

每轮操作记录见[独立日志](docs/change_logs/README.md)，版本摘要见[CHANGELOG](CHANGELOG.md)。历史记录用于审计，当前实现以本页和需求文档为准。

## 基础入口

在项目根目录使用已有依赖的系统Python：

```powershell
python tests/run_regression.py
python -m src.import_data --help
python -m src.source_review --help
python -m src.predict --help
```

数据导入只接受显式提供的本地ZIP和来源记录，不联网。源数据核查须显式提供类别；图片预测只接受JPG/JPEG、PNG、BMP和静态WebP，逐张处理，不递归目录。详细参数查看各入口的`--help`。

当前仍需明确位置框、类别含义、头像来源与裁剪规则。随后建立独立护目镜数据版本和配置，再适配模型与评价。训练、真实数据处理和依赖安装按用户推进执行；当前没有启动这些流程。

护目镜第一版完成前不上传GitHub；保留本地独立提交、日志和初版标签`v0.1.0`。本轮验证结果见[第03轮日志](docs/change_logs/2026-10-14-03-goggles-only.md)。
