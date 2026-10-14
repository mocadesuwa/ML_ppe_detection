# 2026-10-14 第03轮：仅维护护目镜项目内容

- 记录时间：2026-10-14 12:37:39，北京时间（UTC+08:00）。
- 关联提交：本日志与对应修改同次本地提交，编号通过本日志的Git历史查找；不上传GitHub。
- 原因：用户明确任务是护目镜，口罩内容可以移除。本轮直接移除旧任务归档和材料，不继续保留工作区中的旧业务分支。
- 性质：实际操作记录，不是历史补录。历史日志保留审计性质，不作为当前需求或本轮模型结果。

## 删除范围与安全执行

删除前提交为`19dcf5bab9bf36800f85ee7b2c1a6a2945f54e7f`，保留Git历史与`v0.1.0`。

首次计划使用PowerShell递归删除`legacy`、`docs/data_reviews`及指定报告/测试文件，**自动审批拒绝，操作未执行**。理由是整目录永久删除范围可能造成不可逆丢失。没有改用间接命令执行相同的递归删除。

随后先执行`git ls-files`、`git diff --name-only`和未跟踪文件检查，确认以下18份具体文件均已保存在当前Git提交、没有未提交修改，再获准使用**不带`-r`、不带`-f`的`git rm --`逐项清单**移除，命令退出0：

- `legacy/mask/`的8份已跟踪文件：README、baseline.yaml、dataset.yaml、prepare.yaml、source_groups_v1.csv、source_decisions_v1.json、review_raw_dataset.py、derive_dataset.py。
- `docs/data_reviews/`的6份旧数据报告/摘要：3份Markdown与3份JSON。
- `docs/baseline_v0.1.md`、`docs/pretraining_ready.md`、`docs/pretraining_check.json`。
- `tests/test_derivation.py`，对应旧数据专用逐项修补流程。

移除文件后核对仅剩两份前轮测试生成的`.pyc`，逐项删除这两个具体缓存文件，再仅删除经检查为空的目录，未使用递归参数。`legacy`目录已不存在。没有删除或改写`dataset/`、`results/`、`weights/`中的真实文件，也没有删除历史日志或重写Git历史。

已移除材料如需审计可从上述提交按原路径查看，例如`git show 19dcf5b:legacy/mask/README.md`；这不是本轮重新执行旧实验。

## 代码、测试与文档变化

| 范围 | 具体变化 |
| --- | --- |
| `src/source_review.py` | 保留通用VOC只读核查，类别必须显式提供；校验类别表并报告未知对象，保留尺寸/对象、两种坐标约定、重复冲突和来源指纹；输出保护及前后来源不变检查继续保留，人工核查状态明确为false |
| 源数据展示/修补 | 删除写死类别的拼图与少数类展示，以及旧数据专用修补/派生脚本，不将这些规则迁入护目镜流程 |
| 通用测试 | 数据流水线、配置、组合完整性、评价、预测、预检和来源核查的合成类别改为`object_a/object_b/object_c`、`class_a/class_b/class_c`等中性标签，不改真实数据或推断业务类别 |
| 测试契约 | 删除归档训练配置测试；来源核查保留原6项，新增显式类别/未知对象及通用CLI报告2项；必需参数检查增加`--classes` |
| 当前文档 | 重写README、configs说明、框架、需求、数据选择、环境及实验模板；新增`docs/goggles_baseline.md`。AGENTS仅描述当前任务，CHANGELOG新增摘要并维护一个已删除报告的历史导航 |

## 本轮实际验证

- 系统Python在正常本机权限下执行`python tests/run_regression.py`：**141项，10.250秒，OK，退出0**；完整输出在`.cache/regression-2026-10-14-remove-mask.txt`。未跳过用例，使用项目缓存中的临时合成样本及假模型。
- 数量变化明确：前轮146项，移除6项专用派生测试及1项归档配置测试，新增2项通用源数据核查测试，即`146 - 6 - 1 + 2 = 141`。本轮141项不等同于2026-10-08历史记录中的141项。
- 9个模块和2个脚本的`--help`共**11个入口退出0**：import_data、source_review、prepare_dataset、check_dataset、review_dataset、environment、train、evaluate、predict，以及find_related_images和preflight_training。
- `python -m compileall -q src scripts tests`：退出0。
- 扫描33份活动代码、测试、配置与现行文档（排除CHANGELOG和历史日志）：无`mask`、`口罩`或`legacy`残留引用；同时确认`legacy`目录不存在，输出PASS。
- `git diff HEAD --name-only -- dataset results weights requirements.txt requirements-lock.txt`：退出0，无输出；本轮未调用这些真实数据或环境变更流程。
- `git diff --check`、`git diff --cached --check`：退出0，仅有Git的LF/CRLF转换提示，无差异格式错误。
- 全仓库Markdown检查：28份Markdown、50个本地链接和17份日志索引，输出PASS、退出0；UTF-8读取、链接目标、围栏成对和日志索引覆盖通过。
- 本地`origin/main`跟踪引用仍为`9c7f3ccaf4971e0c86335127503f13c8f86cf62a`；`v0.1.0`提交仍为`f6e9a6859893027e3fababe3697838723aa3a875`。未执行push、PR或GitHub写操作。

## 前序回归与限制

继续按既定调用顺序累计回顾坐标转换、来源分组、防泄漏、安全解压、标签、清单/指纹、配置、评价、预测及只读来源核查。移除测试仅对应已明确删除的旧任务功能，通用严格校验、输出保护、失败记录和原件不变检查仍有实际回归覆盖。

数据来源选择、真实头像处理/裁剪、GPU或模型环境复查、真实预测/评价、短跑/正式训练及依赖安装均**未运行**。历史日志和本地原始数据保留，但不被当前配置或代码引用为护目镜数据。

下一步仍需明确位置框、类别含义、头像来源及裁剪规则，再建立独立护目镜数据、配置和最终模型；不把通用核查或回归通过写成护目镜效果已验证。
