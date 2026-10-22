# 2026-10-22 第07轮：旧专用数据清理与护目镜入口整理

- 记录时间：2026-10-22 20:28，北京时间（UTC+08:00）。
- 关联提交：与对应文档修改同次提交，编号通过本日志的Git历史查找。沿用用户授权普通push至`origin/main`，保留初版标签。
- 原因：用户要求去掉口罩相关专用内容及数据，使整体架构清爽、可读。2026-10-14已删除旧专用代码、配置和报告，但当时明确没有删除本地真实数据；本轮补齐该项清理。
- 性质：本轮实际操作与验证。保留旧日志和Git历史用于审计，不改写历史结果。

## 核对与删除范围

源`source.json`指向Kaggle Face Mask Detection；核对853张原图和全部XML对象类别，仅有`with_mask`、`without_mask`、`mask_weared_incorrect`。两版派生清单明确绑定该数据及这三类，旧检查输出和计划也使用同一批文件/类别。活动代码、配置和护目镜来源清单未引用它们。

删除前生成本地`.cache/legacy-cleanup-before.json`，保存逐文件SHA256、数量和字节数，同时记录护目镜保护文件的哈希。删除前解析每个目标的绝对路径，确认位于本工作区并拒绝目录联接/符号链接；使用同一PowerShell中的`Remove-Item -LiteralPath`逐项执行，退出0。

| 已删除目标 | 文件数 | 字节数 |
| --- | ---: | ---: |
| `dataset/raw/face-mask-detection.zip` | 1 | 416,968,083 |
| `dataset/raw/face-mask-detection/` | 1,707 | 417,887,799 |
| `dataset/derived/source_v1/`、`dataset/prepared_v1/`、`dataset/prepared_v2/` | 4,775 | 1,149,454,495 |
| `results/data_check_v1/`、`data_check_v2/`、`data_preparation/`、`preflight_01/`、`scene_review_v1/`、`source_review_2026-10-08/`、`source_review_2026-10-08_v2/` | 97 | 80,416,725 |
| `.cache/baseline-plan.json`、`.cache/smoke-plan.json` | 2 | 2,280 |
| **合计：14个目标** | **6,582** | **2,064,729,382** |

最后检查`dataset/derived/`已空，仅删除这个空父目录。`dataset/raw/.gitkeep`保留通用导入占位；本地删除对象均被Git忽略，没有将真实图片或完整逐图记录加入提交。上述容量约2.06GB（1.92GiB），不是去重后的独立图片数量；其中仅原始来源为853张图，其余包含副本、标注及检查输出。

## 保留与文档整理

- PPEs和SH17是当前护目镜头像的真实来源，属于混合PPE数据。即使图中有人戴口罩、原标注含其他用品，也保留原件和全部标注，不破坏已审核的护目镜样本或源数据哈希。
- 保留`dataset/goggles/`所有版本、`weights/`、分类训练/评价/预测结果、护目镜预检和`docs/data_sources/`，共2,159个文件。两轮模型及最终测试标记、原始来源与审核哈希均保持不变。通用COCO初始权重不属于口罩专用模型。
- 通用代码接受显式类别并使用中性测试数据；头像划分复用来源分组算法，分类预测复用静态输入检查，继续保留这些通用模块。没有修改源代码、测试、YAML实验配置或依赖文件。
- 更新`README.md`与`docs/framework_review.md`，明确当前护目镜主流程、输入及扩展检测接口；重写`configs/README.md`索引，使其指向当前v2_control/v3数据，同时标明对照和历史配置。
- 新增`dataset/README.md`说明当前数据、原图、历史版本和保留原因；同步`AGENTS.md`、`CHANGELOG.md`及本日志索引。历史日志保留原始审计用途。

## 本轮实际验证

- `python -X utf8 .cache/check_legacy_cleanup.py after`：PASS；14个删除目标均不存在，2,159个保护文件目录清单和SHA256与删除前逐项完全一致；本地结果为`.cache/legacy-cleanup-after.json`。
- `python tests/run_regression.py`：**212项，20.207秒，OK，退出0**；完整输出保存在`.cache/regression-2026-10-22-remove-mask-data.txt`。用于确认移除旧本地数据后，通用框架及当前分类流程仍可独立回归，没有跳过用例。
- `python -m src.classification_data check --data dataset/goggles/classification_v3_20261022`：退出0，272张、201/47/24，`integrity_passed=true`、`data_ready=true`、来源组跨集合0；指纹仍为`5859b079a61f07f9436f5a7c75383d9c8854fe73fb98c0211cb4045643ae550c`。
- 对历史`classification_v2_20261022`执行同一只读核查：退出0，191张、133/29/29，来源组跨集合0；指纹仍为`b9176dd923d011a63f7373e126d0997f9959c2e324debc20117140013250e7af`。两次检查均核对原图区域与头像像素，未执行模型前向。
- `python -X utf8 .cache/check_goggles_improvement.py`：PASS；只读核对保存的实验、权重、原测试标记、140张扩充原图及VOC、272张头像、三组运行代码SHA和已有逐图预测，重算已有指标一致，没有新模型前向。输出保存在`.cache/cleanup-goggles-evidence-checks.json`。
- `python -X utf8 .cache/check_cleanup_docs.py`：PASS；45份Markdown、193个本地链接/图片和26份日志索引检查通过；45份活动代码、测试和配置文件不含旧口罩类别或数据路径。`dataset/`仅剩`goggles/`、空`raw/`占位和说明文件；所有文档/配置JSON可解析。输出为`.cache/cleanup-doc-checks.json`。
- `git diff --check`：退出0，无差异格式错误；源代码、脚本、测试、实验YAML、来源清单及依赖文件的差异清单为空。本轮不改写历史提交或标签。

## 前序回归与限制

累计回顾安全解压、坐标与标签、来源分组、防泄漏、清单/指纹、头像裁剪与像素、审核绑定、配置、训练门禁及分类评价/预测契约。旧数据独立于当前护目镜流程，移除后两版冻结数据仍可核查，原实验SHA证据保留。

本轮没有安装依赖、训练、调整模型或阈值，也没有重新评价任何测试集。已有准确率与召回率的取舍仍按[第二轮报告](../goggles_improvement_report.md)解释，目录清理不代表模型效果提高。被删除的本地原图不在Git中，历史日志只能追溯来源和既往操作，不能用于恢复这些文件。
