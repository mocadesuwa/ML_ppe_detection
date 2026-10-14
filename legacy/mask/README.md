# 口罩旧任务归档

本目录保存2026-10-08口罩三类任务的配置、来源组、派生决策和两份专用脚本。它们不属于当前护目镜方案，真实数据仍保留在原位置，没有复制或重标注。

| 原路径 | 当前路径 |
| --- | --- |
| `configs/baseline.yaml`、`dataset.yaml`、`prepare.yaml` | 本目录同名文件 |
| `configs/source_decisions_v1.json`、`source_groups_v1.csv` | 本目录同名文件 |
| `scripts/review_raw_dataset.py`、`derive_dataset.py` | 本目录同名文件 |

归档基线的数据引用、数据配置相对路径和准备配置中的CSV路径已随迁移更新；CSV及逐项决策内容未改。专用脚本调整导入路径，原始核查和派生保护仍由累计回归覆盖。

若只需查看旧计划，在项目根目录运行下列命令；不加载模型、不读取真实数据内容、不创建训练目录：

```powershell
python -m src.train --config legacy/mask/baseline.yaml --dry-run
```

旧核查/派生脚本仅用于历史复现，其默认输入仍是旧口罩数据；当前不执行。2026-10-08的日志和报告按当时的路径解释。原`src.download_data`的固定网络下载已删除，当前通用导入入口是`src.import_data`；历史代码可从Git历史查看。
