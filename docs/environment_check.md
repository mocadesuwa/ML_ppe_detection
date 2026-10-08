# 本机训练环境检查

检查日期：2026-10-07。最初执行查询与导入检查；框架实现阶段已创建项目 .venv，尚未安装训练依赖。CUDA 版 PyTorch 安装请求被拒绝，没有执行安装。用户随后要求暂不训练，本阶段不启动训练。

## 1. 实际观察结果

| 项目 | 实测结果 | 对项目的意义 |
| --- | --- | --- |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU | 可考虑本地小型 YOLO 训练 |
| GPU 显存 | 6144 MiB，约 6 GiB | 先采用小模型和保守 batch，实际用量需测试 |
| NVIDIA 驱动 | 616.92 | 需与所选 PyTorch CUDA 构建配合验证 |
| 默认 python | 3.14.0 | 创建项目环境时核对整套依赖兼容性 |
| PyTorch | 2.12.1+cpu | 当前导入的是 CPU 构建 |
| torch.cuda.is_available() | False | 当前 Python 环境尚不能进行 CUDA 训练 |
| Ultralytics | 未找到模块 | 默认环境缺少 YOLO 软件包 |
| py 启动器登记的其他 Python | 3.9，登记于 E:/vs2022/vsshare/Python39_64/python.exe | 该入口启动失败，不作为项目环境 |

GPU 信息由 nvidia-smi 查询。Python 与 PyTorch 信息来自默认 python 的实际导入输出，不代表系统中所有其他虚拟环境的状态。

检查过程中，CIM 硬件查询不可用，系统内存未核实。默认 python 启动时也出现了定位实际执行文件的提示，但本次版本查询和 PyTorch 导入完成；建立独立环境时应再次核对解释器路径并验证启动过程。

## 2. 环境准备建议

以下为后续准备步骤，项目虚拟环境已创建，训练依赖尚未就绪。

1. 在项目内创建独立的 .venv，并明确记录使用的 Python 解释器。
2. 检查 Windows、Python、PyTorch、torchvision 与 Ultralytics 的兼容版本；采用官方提供的预编译包。
3. 根据 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)选择 Windows、pip 与合适的 CUDA 构建。
4. 安装 Ultralytics，并保存安装后的实际依赖版本。
5. 核实导入路径、PyTorch 版本、CUDA 可用性与 GPU 名称。
6. 做一个小型 GPU 运算检查，再进行 1–3 轮 YOLO 流程检查。

当前官方页面将 Python 3.14 列入 Windows 支持范围，因此不能仅因 Python 较新就断言它不兼容。是否沿用 3.14，还需核查所选的整套依赖和实际安装结果。若需另选 Python，优先考虑在独立项目环境中使用兼容版本。

具体安装命令在确认依赖组合后生成，避免提前写入未经核实的 CUDA 或包版本。当前 CPU 版 PyTorch 是软件环境待处理项，不代表 GPU 硬件损坏。

## 3. GPU 环境完成的判据

- 在项目解释器中能导入 torch 和 ultralytics。
- PyTorch 构建包含 CUDA 支持，torch.cuda.is_available() 返回 True。
- GPU 名称与 RTX 3060 Laptop GPU 对应。
- 一个小型 GPU 运算能完成。
- YOLO 短跑日志确认使用 GPU，并能保存权重。

## 4. 适合本机的初始运行安排

- YOLO11n，输入 640，batch 从 8 开始；若显存不足降到 4 或 2，并记录实际参数。
- 首次 Windows 数据加载设置 workers=0，确保流程正常后再评估速度。
- 短跑检查使用少量数据、1–3 轮；正式 baseline 最多 50 轮。
- 用短跑测到的每轮时间估算正式训练耗时，再结合提前停止和验证开销修正估计。

6 GiB 显存不是 batch=8 一定成功的保证。当前没有实际训练，因此没有耗时、峰值显存或训练吞吐量结果。
