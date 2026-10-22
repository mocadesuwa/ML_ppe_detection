# 第01轮：小规模护目镜候选数据检索与核查

记录时间：2026-10-22 16:53，北京时间（UTC+08:00）。本日志与对应文档、来源清单一起提交，关联提交编号通过本文件Git历史查询，不预填尚未生成的编号。

## 原因与完成条件

用户要求“自己找相关数据集，先不用太大，先不进行训练”，随后授权必要时自行在虚拟环境安装依赖。本轮可检索、少量获取候选原图和原标注并核查，不选定分类/检测路线，不擅自确认护目镜定义或头像框。

完成条件：比较可追溯的相关来源，实际获取小规模原图及原标注；记录版本、许可声明、哈希、数量、真实抽查与局限；不训练、不覆盖原件、不上传原图；同步当前文档并写独立日志。起始工作区干净，HEAD为`5373938`。

## 涉及文件与变化

- `docs/data_selection.md`：四个来源对比、100张原图选择依据、标注组合取样、结构核查、实际视觉疑难与停止点。
- `docs/data_sources/ppes_v7_review_100_20261022.json`：固定revision、来源URL、每图ID/文件名/字节数/SHA256/ZIP CRC32、原标注哈希及核查摘要；不含原图像素，不补造完整压缩包哈希。
- `README.md`、`docs/goggles_requirements.md`、`docs/goggles_baseline.md`、`docs/framework_review.md`：从仅合成数据更新为已有待复核原图，保持真实头像、业务标签和模型未就绪状态；明确177项回归属于2026-10-14历史验证。
- `AGENTS.md`、`docs/environment_check.md`：记录本轮数据准备授权、暂不训练，以及新增依赖仅安装在项目虚拟环境的要求；本次已有Python/Pillow足够，未安装依赖。
- `CHANGELOG.md`、本日志与日志索引：记录本轮摘要及真实验证结果。
- 本地忽略目录`dataset/goggles/source_probe_20261022/`保存来源说明和下载结构探针，`dataset/goggles/ppes_v7_review_100_20261022/`保存100张原图、COCO子集、哈希清单、只读核查及预览；`.cache/fetch_ppes_100.py`和`.cache/verify_ppes_100.py`是本轮一次性取样/核查工具，均不加入Git。未改`src/`、`scripts/`、`tests/`、配置或依赖文件。

## 实际执行与结果

1. 通过公开发布页比较PPEs v7、SH17、CPPE-5、Safety Goggles - PPE v1，原入口与许可声明见数据选择文档。CPPE-5数据卡许可unknown；不把它写成已确认许可。PPEs上游说明提供者为undefined，逐图最初来源仍待核实。
2. 沙箱内`urllib`元数据查询因网络DNS限制失败；获运行权限后查询公开元数据、获取约0.4 MB的官方mini样例并验证HTTP Range。没有执行第三方加载脚本或安装数据集SDK。
3. `python -X utf8 .cache/fetch_ppes_100.py --metadata-only`退出0；读取ZIP目录和完整原始COCO标注1,073,008字节，原train为6,473张。`python -X utf8 .cache/fetch_ppes_100.py`退出0；固定revision `90d2c6da950a6168fcee20ec69e194a034f44eef`，仅从原train以种子42取100张（原标注组合40/40/20）。原图13,696,425字节，100张获取过程读15,157,085字节；未下载完整train.zip，逐成员长度与CRC32均核对。
4. `python -X utf8 .cache/verify_ppes_100.py`初次退出1：一次性工具误用了当前Pillow不存在的`get_flattened_data`。改用已有`getdata`后重跑退出0；未升级Pillow。100张全部1280×720，SHA256、解码/尺寸、目录数量和COCO引用一致；146框（goggles 68、no_goggles 78），非法框0；字节及解码像素完全重复组均为0。25个眼部框短边小于32像素；整图dHash距离≤8有831对相似候选，未当作确认重复或自动来源组。
5. 实际打开三张分层抽查图查看12张（每层4张），原图和最多两个眼部放大区同屏；另早期12张临时预览重叠3张，候选池累计查看21张。原图765、170、538的细框眼镜类型不确定，715、5895有头顶/额头位置疑难，多个固定机位和相近时间帧；逐项编号见来源清单。没有全量人工质量审核，不把生成拼图等同审核完成。
6. 本日志建立后执行内联Python检查，退出0：32份Markdown的UTF-8与围栏、72个本地链接、20份日志索引覆盖全部通过；来源清单100条与本地下载清单一致，全部原图及原始/子集COCO的SHA256一致。证据为`.cache/small-dataset-doc-checks.json`。`git diff --check`退出0，仅有AGENTS.md的LF/CRLF提示。
7. `git diff --name-only HEAD -- src scripts tests configs requirements.txt requirements-lock.txt weights`无输出；`git ls-files -- dataset .cache .venv weights`仅有`dataset/raw/.gitkeep`及`weights/.gitkeep`，原图和一次性脚本均通过`git check-ignore`确认隔离。未将真实图片、预览、缓存、环境或权重加入版本管理。

## 前序回归与边界

本轮未修改项目程序，`python tests/run_regression.py`未运行；按项目“仅文档变更执行文档检查”的约定，不把2026-10-14的177项通过补写为本次结果。既有裁剪、完整性和分组代码保持不变，未以COCO候选数据调用仅支持VOC的`src.source_review`。

未运行真实头像生成、重标注、冻结train/val/test、模型加载/预测、无梯度预检、短跑、正式训练、模型评价、依赖安装或GPU检查。正式业务类别、眼镜类型和裁剪规则仍待明确；相似候选不能当作人物识别或序列分组结果。原件保留，状态为`raw_review_pool`、`training_ready=false`、`visual_review_completed=false`。

## 版本管理与下一步

按已有授权将文档和来源清单以独立提交普通push到`origin/main`，不强推、不改初版`v0.1.0`、不上传原图/预览/权重。本日志在提交前记录，不预填尚未执行的同步结果，回执保存在本地`.cache/`。

提交前`git fetch origin`退出0，`origin/main...HEAD`为0落后/0领先，本地与远端均为`5373938dba5167890504955994cff78a48bee7aa`。未改写历史或标签。

下一步结合样本明确普通眼镜、额头佩戴与无法判断规则，核查原始来源/序列，再确定显式头部框与业务标签。若该工业监控来源不适合最终定义，先补查小型备选，不将这100张强行宣布为护目镜训练集。
