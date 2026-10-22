# 护目镜任务配置

用户已确认头像二分类，只输出佩戴状态、不输出框。当前选定配置是[goggles_classification_v2_control.yaml](goggles_classification_v2_control.yaml)，绑定272张v3数据、ImageNet初始权重及匹配预检；实际结果见[改进报告](../docs/goggles_improvement_report.md)。配置中的v2指第二轮模型方案，数据版本为v3。

| 配置 | 用途 | 数据版本 |
| --- | --- | --- |
| [v2_control](goggles_classification_v2_control.yaml) | 当前选定方案：随机抽样、按训练类别频次加权 | v3，272张 |
| [v2_groups](goggles_classification_v2_groups.yaml) | 已完成的来源组均衡抽样对照，未选中 | v3，272张 |
| [v2_uniform](goggles_classification_v2_uniform.yaml) | 已完成的来源组均衡抽样、不加类别权重对照，未选中 | v3，272张 |
| [v2_smoke](goggles_classification_v2_smoke.yaml) | 第二轮两轮短跑，验证更新、冻结与重载 | v3，272张 |
| [v1](goggles_classification_v1.yaml) | 第一轮基线，保留模型来源和旧评价绑定 | v2，191张 |
| [smoke_v1](goggles_classification_smoke_v1.yaml) | 第一轮短跑历史记录 | v2，191张 |

```powershell
python -m src.classification_data check --data dataset/goggles/classification_v3_20261022
python -m src.train_classification --config configs/goggles_classification_v2_control.yaml --dry-run
```

分类训练入口默认与`--dry-run`均只检查和打印计划。真实训练必须显式提供`--start`并通过匹配预检；上表已有实验均已完成，输出目录不能覆盖。新实验需新配置、新预检和新的独立保留评价。本次目录清理没有重新训练或测试模型。

[head_crops.example.json](head_crops.example.json)是显式头像框格式示例，中性标签仅供演示。当前实际框见[头像清单v3](../docs/data_sources/goggles_v3_head_plan.json)，审核见[v3质量记录](../docs/data_sources/goggles_v3_quality_review.json)；二者均绑定冻结数据。来源图像、头像、划分目录及权重只在本地，未随Git提交。
