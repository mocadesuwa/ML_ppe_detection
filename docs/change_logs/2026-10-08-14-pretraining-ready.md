# 14：CUDA、初始权重、无训练预检与结尾累计回归

- 记录时间：2026-10-08 23:31:02，北京时间（UTC+08:00）。
- 关联提交：本日志与对应修改同次提交，通过 Git 历史查找编号。
- 原因：按用户授权一直完成到训练前一步，并按最新要求把前序累计回归统一放在结尾。
- 文件：requirements 与锁定文件、`scripts/preflight_training.py`、`tests/test_preflight.py`、`tests/test_entry_configs.py`、初始权重/预检 JSON、训练前报告、README、环境/框架/方案/数据选择文档、CHANGELOG、本日志及索引。本地真实产物继续被 Git 忽略。

## 环境安装与权重

官方 CUDA index 实查及官方版本表确定 torch 2.11.0 / torchvision 0.26.0 / cu128 / cp314 Windows 组合。最初 pip 下载 2.77 GB wheel 一直为零字节，检查官方 HEAD 和 1 MiB 范围请求后，确认该任务 pip 进程并中止；同一官方 URL 分段下载 331 块，按大小与官方 SHA-256 逐项验收并拼接。没有把中止的 pip 尝试记作成功。

实际安装命令：

```powershell
.\.venv\Scripts\python.exe -m pip install '.cache/pip_wheels/torch-2.11.0+cu128-cp314-cp314-win_amd64.whl' torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128 --only-binary=:all: --report .cache/install_reports/pytorch-cu128.json
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --index-url https://pypi.org/simple --only-binary=:all: --report .cache/install_reports/project-dependencies.json
.\.venv\Scripts\python.exe -m pip install 'scipy>=1.16' --index-url https://pypi.org/simple --only-binary=:all: --report .cache/install_reports/scene-dependencies.json
```

均退出 0。SciPy 为已使用的 pHash 工具补齐依赖；首次项目安装时尚未写入该要求，所以后续单独安装并保存报告。实际 PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128、Ultralytics 8.4.174、SciPy 1.18.1。没有更新系统 Python、NVIDIA 驱动、CUDA toolkit 或 pip。

`pip check` 退出 0：No broken requirements found。`pip freeze` 保存 41 包版本；torch 的本机 wheel URI 转为可移植版本锁定，并保留官方 CUDA index。项目缓存设置指向项目目录，关闭 Ultralytics 同步及可选外部集成，未上传数据。

官方 GitHub v8.4.0 release API 取得 YOLO11n 资产，下载 5,613,764 字节，发布方 digest 和本地 SHA-256 均为 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`。先暂存校验再发布权重，来源记录提交、模型文件不提交。

## 本阶段实际验证

- `.venv` 执行 `python -m src.environment --output results/environment_pretrain.json`：CUDA True、RTX 3060 Laptop 6 GiB、CUDA 12.8、小矩阵运算通过。安装中途的 `environment_pretrain_torch.json` 仅为 torch 阶段观察，最终以完整报告为准。
- 系统 Python 单独运行 `test_preflight.py`：5 项新增门禁测试通过，0.286 秒；过期核查、缺标签/本地权重、已有输出及写入冻结数据内都被拒绝，不生成 readiness 报告。
- `.venv` 执行 `scripts/preflight_training.py`：真实加载扫描 510 张，无损坏；关闭增强，batch=8/640 只做 eval/inference_mode/AMP 前向，输出 `[8,84,8400]` 有限；CUDA NMS 通过；梯度 0；权重和数据指纹前后未变。前向峰值 allocated 159.962890625 MiB，不是训练或显卡总显存。
- 预检生成的 Ultralytics `train.cache` 属于派生标签缓存，图像、TXT、清单、划分与质量记录指纹未变；缓存路径确在项目内。终端部分库日志对中文路径显示简化，实际文件与 JSON 路径核对正常。
- 两种训练 dry-run、预检 help、`compileall -q src scripts tests` 退出 0。`smoke_01` 与 `baseline_01` 均不存在；没有调用 YOLO 训练或优化器。

## 结尾累计回归

1. 首次 `python tests/run_regression.py`：141 项，9.607 秒，1 项失败。默认训练计划测试仍写死 `dataset`，与本轮已冻结的 `dataset/prepared_v2` 指针不符；其余 140 项通过。该失败没有忽略。
2. 将该测试改为确认当前冻结版本及其训练目录，保留原有参数、权重路径与三类检查。未为通过测试改回旧数据。
3. 再运行系统 `python tests/run_regression.py`：141 项，6.709 秒，全部通过；输出保留 `.cache/regression-final-system.txt`。
4. 本轮 `.venv` 为新安装的实际运行环境，因此再用 `.venv/Scripts/python.exe tests/run_regression.py` 验证：141 项，6.864 秒，全部通过；输出保留 `.cache/regression-final-venv.txt`。两轮均获正常本机临时目录权限，不跳过重命名测试。

此前日志 12、13 的“回归留到最后”已在本轮落实。累计包含此前 130 项、6 项派生与5项预检门禁；假模型和合成样本仍不代表真实效果。最终又核对原目录 1,707 个文件快照与原 ZIP SHA-256，均未变；初版 `v0.1.0` 仍指向 `f6e9a68`。

收尾文档检查：26 份 Markdown 的本地链接全部存在，日志已填实际北京时间；`git diff --check` 无错误。逐项对照当前安装元数据，41 个可移植版本锁定一致；系统 torch 发行版本仍为 2.12.1。修正日志索引新增行之间的空行，使14轮记录保持同一张表。

## 停止点与限制

已完成当前授权的全部训练前工作，停止在首次短跑前。保持三类但 test 的 incorrect 仅 5 个目标；181 张摆拍系列整体在 val，需解释画面分布差异。未验证反向传播、优化器、增强、实际训练显存、训练吞吐量、短跑权重保存、任务指标或实验室效果。

下一步由用户明确启动 GPU 短跑；本轮不自动推进训练。
