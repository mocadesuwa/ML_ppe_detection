# 08：共享配置与训练计划预检

- 记录时间：2026-10-08 15:56，北京时间（UTC+08:00）；非提交时间。
- 关联提交：本日志随代码一起提交；提交编号可用 `git log -- docs/change_logs/2026-10-08-08-entry-configs.md` 查询。
- 文件：`src/common.py`、`src/train.py`、`tests/test_entry_configs.py`、`configs/baseline.yaml`、`README.md`、`CHANGELOG.md`、`docs/framework_review.md`、本日志与索引。

## 原因与具体变化

数据处理测试顺序中的五项检查已完成，继续检查运行入口最早使用的共享配置和训练计划。旧实现会让 YAML 重复字段静默覆盖前值，也会接受小数或布尔值冒充训练轮数、batch 等参数；缺字段、空路径和错误类型容易产生底层异常。错误设备格式还会在校验前导入 PyTorch。

1. YAML 读取采用安全加载器，在合并前检查显式重复键，包含 `0` 与 `false` 这种会互相覆盖的键；保留普通别名、合并覆盖、等号键及 UTF-8 BOM 兼容。文件不存在、无法读取、编码错误及 YAML 语法错误包含文件路径。
2. 共享路径校验拒绝空字符串与非字符串。数据配置路径仍相对 YAML/数据根目录解析，训练的 data/model/project 仍相对项目根目录解析；baseline 配置补充注释，没有改变既有参数。
3. 训练计划检查必需字段与最终有效参数：epochs/batch/imgsz 为正整数，patience/workers/seed 为非负整数，布尔开关类型明确，cache 为布尔值或 ram/disk，fraction 为 `(0, 1]` 有限数值。极大整数 fraction 也给出字段错误，不因浮点转换溢出而崩溃。
4. 保持 YAML、smoke 设置、CLI 覆盖的优先级；有效 CLI 覆盖后再检查参数，不改写配置文件。project 已是普通文件时提前拒绝。
5. 实验名称在创建目录前检查，拒绝空值、非法字符及 Windows 保留设备名；设备格式支持 auto/cpu/一个非负编号。无效设备和 CPU 选择均不在此阶段导入 PyTorch，CUDA 可用性仍在实际运行时查询。
6. 训练入口的配置错误使用 argparse 提示退出，状态码 2。dry-run 继续只打印计划，跳过真实数据内容检查、缓存及实验目录创建和模型加载。
7. 更新当前验证状态；日志索引补充上轮已确认的提交时间与编号。

## 实际验证与前序回归

使用现有系统 Python 3.14、Pillow 12.0.0 与 PyYAML 6.0.3，没有安装依赖。

- 首次运行 `python -m unittest discover -s tests -p test_entry_configs.py -q` 时，沙箱阻止临时目录创建/清理，20 项测试报告 40 个环境错误，未获得逻辑验证结果。经授权在沙箱外重跑同一命令，修正前复现 113 个失败、37 个错误（含子用例）。这次只针对新增 20 项，没有把前序测试写作已重新执行。
- 修正后执行 `python tests/run_regression.py`：75 项累计测试通过，耗时 3.656 秒。
- 复查发现 YAML 等号键及超大 fraction 的兼容/溢出边界，补齐处理与子用例后再次运行累计入口：`Ran 75 tests in 3.646s`，`OK`。这是最终代码的验证结果；75 项包含前序 55 项与新增 20 项。
- 新测试覆盖非法配置和路径、重复键、YAML 合并、中文及空格路径、默认计划、smoke/CLI 优先级、设备与名称校验、短错误提示。dry-run 测试拦截 Torch/Ultralytics/Pillow 导入、数据内容检查与缓存创建，并核对临时目录内容没有新增。
- 实际命令行执行 `python -m src.train --dry-run` 和 `python -m src.train --smoke --dry-run`，均退出 0 并打印计划；没有执行短跑或训练。
- 实际执行 environment、download_data、prepare_dataset、check_dataset、review_dataset、train、evaluate、predict 共 8 个模块的 `--help`，均退出 0。
- 最终累计回归样本由入口放在 `.cache/tests/` 临时目录并自动清理；前序坐标、分组、解压、清单/指纹、标签校验全部通过。

提交前本地文档链接检查通过：12 份文档的本地链接均存在，8 份独立日志齐全；`git diff --check` 通过。

## 限制与下一步

本轮没有修改真实数据、准备真实划分、下载权重、安装依赖或运行模型。dry-run 验证的是项目配置与计划，不代表 CUDA 可用、数据质量合格或真实 YOLO 参数全部兼容；扩展模型参数仍需后续验证。项目 `.venv` 的依赖准备状态保持不变。

共享设备/名称校验也用于评价和预测，但这两个入口的其余参数、记录一致性和错误提示仍待专项检查。下一轮按流程继续评价入口，再检查预测入口，每轮保持累计回归与独立日志。
