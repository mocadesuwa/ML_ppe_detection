# 2026-10-14 第02轮：旧任务清理与静态图片基础整理

- 记录时间：2026-10-14 12:22:18，北京时间（UTC+08:00）。
- 关联提交：本日志与对应修改同次本地提交，编号通过本日志的Git历史查找；不上传GitHub。
- 原因：用户推进按新需求修改原框架、适当删除无用内容并保持代码清晰。位置框要求仍待确认，本轮处理不依赖最终模型路线的清理，不擅自确定护目镜类别或裁剪参数。
- 本轮性质：实际代码与文档调整，不是历史补录。当前检查结果与2026-10-08历史证据分别记录。

## 文件范围与具体变化

| 范围 | 文件与变化 |
| --- | --- |
| 口罩专用内容 | 五份`configs/`配置/决策及`scripts/review_raw_dataset.py`、`derive_dataset.py`移入`legacy/mask/`；修正相对数据路径、配置引用、CSV路径和模块导入，新增归档说明 |
| 本地导入 | `src/download_data.py`改为`src/import_data.py`，删除固定口罩URL、网络下载、进度与超时分支；要求显式ZIP、目标目录和来源URL，保留安全解压。来源元数据写到同级`<目录名>.source.json`，不覆盖ZIP中的同名原件；前后核对ZIP哈希，已有记录不覆盖 |
| 显式配置 | `src/prepare_dataset.py`、`check_dataset.py`、`review_dataset.py`、`train.py`、`evaluate.py`及`scripts/preflight_training.py`要求显式配置/数据；预测、评价要求显式权重。`scripts/find_related_images.py`要求显式来源，取消旧派生目录默认值 |
| 图片入口 | `src/predict.py`只接受JPG/JPEG、PNG、BMP和静态WebP；在加载模型前检查空文件、可校验性与动画。目录只筛选直接图片，逐张交给后端，不递归或传入混合目录 |
| 结果格式 | 预测`format_version=2`，使用`image_index`、`image_count`，移除`frame_index`、`source_frame_index`、`images_or_frames`。保留逐行`.part`、来源/权重绑定、框与类别检查、重复/遗漏及失败/中断记录 |
| 测试 | `tests/test_data_pipeline.py`、`test_derivation.py`、`test_source_review.py`更新导入；`test_entry_configs.py`显式检查归档计划；`test_prediction_entry.py`改为图片语义并新增损坏图片/视频和APNG拒绝检查；新增`test_task_cleanup.py`覆盖必需参数、来源记录与原件保护 |
| 文档与忽略 | 精简README和`docs/framework_review.md`；更新AGENTS、需求与CHANGELOG；新增`configs/README.md`，预留`dataset/goggles/`忽略规则；维护旧来源报告的脚本链接并注明迁移，旧结论不变；更新本日志及索引 |

仍保留检测后端及检测数据校验能力供后续适配；这不是已经选定护目镜检测路线。未新增自动头部检测、头像裁剪参数或通用双模式框架。旧类名在归档及历史合成测试中保留原义，没有改名当作护目镜标签。

## 累计回归：真实执行过程

1. 系统Python在沙箱内执行`python tests/run_regression.py`，输出前11项通过后长时间未完成；没有完整结果，不能记为通过。原输出保存在`.cache/regression-2026-10-14-cleanup-first.txt`。
2. 沙箱内查询该测试进程也被拒绝，命令退出1。获正常本机权限后，核对PID 24236的命令行为本轮回归并终止该未完成进程，没有终止其他Python服务；不将未完成归因于未经证明的具体代码错误。
3. 在正常本机权限下重新执行同一`python tests/run_regression.py`：**146项，11.521秒，OK，退出0**。完整输出在`.cache/regression-2026-10-14-cleanup-local.txt`。没有跳过目录重命名或其他用例。
4. 核对绝对路径位于项目`.cache/tests/`后，删除本轮被中止运行遗留的`regression-hezqzoho`临时合成样本，退出0；成功运行自身正常清理，最终测试缓存无该遗留目录。输出日志保留。

146项包含原141项和新增5项。原视频流用例改为两张静态图片逐张写入并关闭结果流的用例，另扩展混合目录筛选验证；不是简单删除原失败路径测试。临时合成样本和假检测模型不能证明护目镜效果。

## 其他实际验证

- 8个`python -m src.<模块> --help`：import_data、prepare_dataset、check_dataset、review_dataset、environment、train、evaluate、predict；另4个脚本`--help`：find_related_images、preflight_training及两份归档脚本。**共12个帮助入口退出0**，没有运行模型或数据流程。
- `python -m compileall -q src scripts legacy tests`：退出0。
- 比较Git中旧CSV/决策JSON与归档文件，规范化Git的CRLF/LF后内容完全一致，检查脚本输出PASS。真实数据、权重和环境锁定文件未改。
- `git diff --name-only HEAD -- dataset weights results requirements.txt requirements-lock.txt`：退出0，无输出。此检查仅覆盖Git跟踪差异；“未处理真实数据”的依据还包括本轮命令仅操作代码、文档和临时合成样本，没有调用真实数据流程。
- `git check-ignore --no-index dataset/goggles/synthetic-example.png`：退出0，输出该路径；这是虚拟路径检查，未创建头像文件。
- `git diff --check`：退出0，仅有按Git配置转换LF/CRLF的提示，无差异格式错误。
- 首次文档检查：32份Markdown、65个本地链接和16份日志索引，退出1；发现旧来源报告仍链接已迁移脚本。维护该链接并注明2026-10-14迁移后，复查同样32份Markdown、65个本地链接和16份日志索引，输出PASS、退出0；UTF-8读取、链接存在、围栏成对和日志索引覆盖通过，未修改旧核查结论。
- `git rev-parse 'v0.1.0^{commit}'`：仍为`f6e9a6859893027e3fababe3697838723aa3a875`；本地`origin/main`跟踪引用仍为`9c7f3ccaf4971e0c86335127503f13c8f86cf62a`，不是远程状态查询。本轮没有push、PR或其他GitHub写操作。

## 对前序修改的回顾

按原入口顺序累计回顾坐标转换、分组、防泄漏、安全解压、组合/指纹、标签、配置、评价、预测和来源核查。归档函数及显式旧训练计划仍纳入回归，前序严格校验、原件保护和失败记录没有为简化代码被删除。

第01轮的“先明确范围、框要求待确认、不上传”约定仍保留；用户本轮已推进代码整理，AGENTS和需求状态同步更新。历史数据核查与GPU记录没有重写为本次结果。

## 限制与下一步

真实数据下载/导入、裁剪、标注、划分、GPU预检、真实模型预测/评价、短跑/正式训练、依赖安装均**未运行**。本轮使用系统Python已有依赖，未改变环境或训练参数锁定。

下一步明确输出位置框、类别含义、头像来源和裁剪规则，再建立独立护目镜数据与任务配置。图片入口只能检查格式，不能自动判定一张图是不是合格头部头像；保留的检测输出不能直接当作最终佩戴判断。护目镜第一版的真实模型和指标仍未完成，不上传GitHub。
