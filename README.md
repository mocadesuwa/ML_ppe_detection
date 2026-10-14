# 实验室护目镜佩戴识别

机器学习课程实践项目：对**静态头部裁剪头像**识别护目镜佩戴情况。是否需要位置框仍待确认，最终分类/目标检测路线、类别及裁剪细则见[需求清单](docs/goggles_requirements.md)。

当前是**头像数据准备版**：已支持显式头部框裁剪、来源组和哈希追溯，尚无已核查的护目镜真实数据、训练模型或指标。图片入口目前保留检测后端，最终模型尚未确定。

## 当前目录

| 位置 | 用途 |
| --- | --- |
| [src/](src/) | 本地数据导入、显式类别源数据核查、数据准备、训练/评价与图片预测 |
| [scripts/](scripts/) | 相似图片候选与无梯度预检，须显式指定输入 |
| [configs/](configs/README.md) | 护目镜配置位置，任务契约明确后建立 |
| [tests/](tests/) | 使用中性标签、临时合成数据和假模型的通用回归 |
| [docs/framework_review.md](docs/framework_review.md) | 当前代码职责和图片结果格式 |
| [docs/goggles_baseline.md](docs/goggles_baseline.md) | 护目镜第一版实施方案与待确认项 |
| [docs/head_crops.md](docs/head_crops.md) | 头像清单、像素坐标约定、裁剪命令与来源记录 |
| [docs/data_selection.md](docs/data_selection.md) | 护目镜数据选择、头像和划分规则 |
| [docs/experiment_record.md](docs/experiment_record.md) | 护目镜实验记录模板，未填写实测结果 |

每轮操作记录见[独立日志](docs/change_logs/README.md)，版本摘要见[CHANGELOG](CHANGELOG.md)。历史记录用于审计，当前实现以本页和需求文档为准。

## 基础入口

在项目根目录使用已有依赖的系统Python：

```powershell
python tests/run_regression.py
python -m src.import_data --help
python -m src.source_review --help
python -m src.prepare_head_crops --help
python -m src.predict --help
```

数据导入只接受显式提供的本地ZIP和来源记录，不联网。源数据核查须显式提供类别；图片预测只接受JPG/JPEG、PNG、BMP和静态WebP，逐张处理，不递归目录。详细参数查看各入口的`--help`。

运行无模型的合成头像演示（使用新目录名）：

```powershell
python -X utf8 scripts/demo_head_crops.py --output .cache/head_crop_demo
```

本轮153项累计回归通过，真实命令行已用合成图片验证裁剪和dry-run。演示不包含真实人物或佩戴标签，不能当作护目镜识别效果。

当前仍需明确位置框、类别含义、头像来源与裁剪规则。随后建立独立护目镜数据版本和配置，再适配模型与评价。训练、真实数据处理和依赖安装按用户推进执行；当前没有启动这些流程。

用户已明确要求本轮完善后同步GitHub，本次以头像数据准备版上传；仍保留本地独立提交、日志和初版标签`v0.1.0`。这不表示完整模型已完成。本轮结果见[第04轮日志](docs/change_logs/2026-10-14-04-head-crops.md)。
