# 护目镜数据来源与复核记录

当前正式版本为v2头像/划分与分类准备版：

- [PPEs原100张来源](ppes_v7_review_100_20261022.json)保留首次下载哈希与抽查。
- [SH17作者原80张来源](sh17_original_80_20261022.json)固定Kaggle版本1，逐图/原VOC可追溯；镜像仅定位候选。
- [v1复核](goggles_v1_review.json)、[v1头像框](goggles_v1_head_plan.json)是198张试裁剪过程记录，不是当前训练输入。
- [v2最终复核](goggles_v2_review.json)、[v2头像框](goggles_v2_head_plan.json)记录226个候选的决定及191张最终头像。
- [v2质量记录](goggles_v2_quality_review.json)绑定精确头像指纹、全部保留编号、视觉/分组核查与训练来源覆盖规则。
- [v2固定划分成员](goggles_v2_split_assignments.json)保留133/29/29的编号、来源组、标签/哈希及来源覆盖调整，不公开本机绝对路径；保留原本地划分清单哈希作对照。

原图和头像不随Git上传。完整操作、限制和数据目录见[处理方案](../data_processing_plan.md)，当前数据与预检指纹见[训练前状态](../pretraining_status.json)。
