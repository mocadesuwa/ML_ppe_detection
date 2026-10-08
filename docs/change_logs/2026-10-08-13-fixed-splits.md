# 13：画面分组、固定划分与最终数据核查

- 记录时间：2026-10-08 22:47，北京时间（UTC+08:00）。
- 关联提交：本日志与修改同次提交，通过 Git 历史查找编号。
- 原因：哈希阈值 3 遗漏画面裁切及摆拍序列；在训练前完成实际分组与各集合核查。
- 文件：`scripts/find_related_images.py`、两个数据 YAML、`configs/source_groups_v1.csv`、画面组 JSON、最终数据摘要/报告、README、框架说明、CHANGELOG、本日志及索引。

## 操作与结果

1. `python scripts/find_related_images.py --output results/scene_review_v1`：794 张、35 对、6 张拼图。系统 Python 的 numpy/scipy 用于候选 pHash，不训练模型。
2. 读取该候选记录，用系统 Python 扩大距离至 dHash≤12/pHash≤16，生成另 176 对的局部拼图，实际全部查看；再用 Pillow 生成并实际浏览全 794 张小图，补充共同背景/布景组。脚本支持用相同距离复现候选，分组由审查明细决定，不自动接受相似哈希。
3. 初次 `python -m src.prepare_dataset` 与 `python -m src.check_dataset --output results/data_check_v1 --samples 30`：555/159/80、全量程序检查通过，实际查看 90 张标注图；发现摆拍系列漏组，因此不接受该版质量核查。
4. 扩大人工组后再执行 prepare/check，输出新版本 `prepared_v2`：510/206/78，579 组，三类目标分别为 1983/439/39、565/121/11、286/61/5，全量检查无错误。保留初次版本，未覆盖冻结数据。
5. 实际查看最终 90 张（各集合 30 张）标注图，再执行 `python -m src.review_dataset --reviewer Codex --notes <实际核查文字> --license-source https://www.kaggle.com/datasets/andrewmvd/face-mask-detection`，当前指纹质量记录写入成功。完整文字保存于摘要及质量记录。
6. 系统 Python 重新调用 read_samples/group_samples/split_groups：与清单逐项一致；所有人工组只在一个集合，跨集合数 0；require_review 接受当前指纹；原目录快照与派生记录一致。摘要保存真实样本名及哈希。

并按已确认提交时间修正上一日志的记录分钟与索引时间；原误填分钟在上一日志末尾保留说明。

## 前序回归与限制

按用户本轮要求，前序累计回归尚未运行，留到全部训练前准备完成后。上述数据检查是本阶段实际检查，不是历史测试补录。

181 张相近摆拍集合整体被分到验证集，导致图片比例偏离 70/20/10，但目标总数比例接近目标；文档已说明验证画面分布变化。人工组是保守工程推断，来源元数据仍不完整，不能宣称全部来源泄漏已排除。少数类 test 仅 5 个目标；原始发布者其他类别标签未全量人工复标，保留新闻/广告实拍等复杂画面。

下一步：完成本地 CUDA 依赖、初始权重和实际但不训练的加载/前向预检，然后统一累计回归。当前未训练、未改原件，环境安装单独记录。
