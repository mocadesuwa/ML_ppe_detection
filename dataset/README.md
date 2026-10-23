# 护目镜数据目录

当前输入是静态头部头像，仅有`not_wearing_goggles`和`wearing_goggles`两类。当前数据和原件均在被Git忽略的`goggles/`内，来源及复核记录见[来源索引](../docs/data_sources/README.md)。

| 本地目录（相对`dataset/goggles/`） | 用途 |
| --- | --- |
| `classification_v3_20261022/` | 当前分类输入，train/val/新test为201/47/24，共272张 |
| `heads_v3_20261022/`、`splits_v3_20261022/` | 当前头像和固定来源组划分，供完整性核查追溯 |
| `ppes_v7_review_100_20261022/` | 首轮100张PPEs原图和原标注 |
| `sh17_original_60_20261022/`、`sh17_original_20_extra_20261022/` | 首轮80张SH17作者原图和原VOC |
| `sh17_expansion_140_20261022/` | 扩充140张SH17原图和原VOC，复核后保留110张头像 |
| `heads_v2_20261022/`、`splits_v2_20261022/`、`classification_v2_20261022/` | 第一轮191张冻结数据，绑定旧模型和旧29张测试记录 |
| `heads_v1_20261022/`、`splits_v1_20261022/`、`classification_v1_20261022/` | 首轮试裁剪与导出过程，供复核追溯 |
| `processing_v1_20261022/`、`source_probe_20261022/`、`sh17_probe_20261022/` | 复核拼图、来源元数据和下载结构探针 |
| `construction_review_100_20261023/`、`construction_probe_20261023/` | 新来源候选及核查探针；存在重复/增强和眼镜语义问题，未纳入当前数据 |

`raw/.gitkeep`仅保留通用本地ZIP导入的空目录。旧口罩专用压缩包、原图/标注、派生副本及其检查输出已清理，详见[清理日志](../docs/change_logs/2026-10-22-07-remove-mask-data.md)。

PPEs和SH17是护目镜样本的实际来源。原图中可能同时出现其他防护用品，原标注也保留发布方类别；这些文件参与护目镜裁剪及哈希核查，不能按其他类别词汇删除。原件、冻结头像及审核清单保持一致；后续改框、改标签或改划分时建立新版本。

Construction-PPE候选属于隔离核查池，不是新的训练版本。只完成了来源及逐项状态候选审核，现有训练仍为272张v3；完整决定见[来源索引](../docs/data_sources/README.md)。

当前完整性核查：

```powershell
python -m src.classification_data check --data dataset/goggles/classification_v3_20261022
```

此命令只核查数据，不执行模型前向、训练或重新评价测试集。训练入口与配置见[配置说明](../configs/README.md)。
