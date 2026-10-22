# 护目镜任务配置

用户已确认头像二分类，只输出佩戴状态、不输出框。当前配置为[goggles_classification_v1.yaml](goggles_classification_v1.yaml)，绑定最终数据指纹、ImageNet初始权重完整哈希及预检报告。规则和结果见[数据处理方案](../docs/data_processing_plan.md)。

```powershell
python -m src.classification_data check --data dataset/goggles/classification_v2_20261022
python -m src.train_classification --config configs/goggles_classification_v1.yaml --dry-run
```

分类训练入口默认与`--dry-run`均只检查和打印计划；未来真实训练必须显式提供`--start`，并已有当前配置/数据/权重匹配的无梯度报告。本轮没有启动训练。不要使用通用检测入口`src.train`训练这套头像分类数据。

[head_crops.example.json](head_crops.example.json)仍是显式头像框格式示例，中性标签不是业务标签。实际最终框见[头像清单v2](../docs/data_sources/goggles_v2_head_plan.json)，实际质量记录绑定头像指纹。来源图像、头像、划分目录及初始权重只在本地，未随Git提交。
