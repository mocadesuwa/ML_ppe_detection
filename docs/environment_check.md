# 本机训练前环境与复现

更新：2026-10-08，北京时间。用户授权完成训练前准备；记录实际安装和不训练的检查。2026-10-07 的 CPU 检查与未获准安装属于历史状态，本轮已完成项目 `.venv` CUDA 准备，系统 Python 未安装或替换依赖。

## 实际环境

| 项目 | 当前结果 |
| --- | --- |
| 系统 / 项目 Python | Windows 11 x86_64 / Python 3.14.0 |
| GPU / 驱动 | RTX 3060 Laptop，6144 MiB，616.92 |
| PyTorch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| CUDA 构建 / 可用 | 12.8 / True |
| Ultralytics | 8.4.174 |
| Pillow / PyYAML / SciPy | 12.3.0 / 6.0.3 / 1.18.1 |
| pip check | No broken requirements found |

使用 [PyTorch 官方版本配对](https://pytorch.org/get-started/previous-versions/)中的 2.11.0/0.26.0 CUDA 12.8 组合，官方 index 实际提供 cp314 Windows 二进制包；Python 范围见[官方安装页](https://pytorch.org/get-started/locally/)。未安装 CUDA toolkit、更新驱动或更改系统 Python。

第一次 pip 大包下载持续零字节，已中止该任务；随后从同一官方文件可恢复分段下载，核对 2,770,971,557 字节与官方响应头 SHA-256 `d6c21797ff75271b4fbdd905e2d703be4ecea5ea5bbdde4d1c201e9c71bc411d` 后从本地 wheel 安装。报告保存在 `.cache/install_reports/`，未把中止尝试记作成功。

## 新环境复现

本机已经安装，无需重复运行。以下供新的 Windows x86_64/Python3.14 环境复现：

```powershell
python -m venv .venv
$env:PIP_CACHE_DIR = Join-Path (Get-Location) '.cache/pip'
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128 --only-binary=:all:
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt --only-binary=:all:
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m src.environment --output results/environment_new.json
```

锁定文件含41个实际包版本；本机 torch 从已校验 wheel 安装，freeze 的本机 file URI 已替换为可移植的 `torch==2.11.0+cu128`。文件含官方 CUDA extra index，其他包按精确版本解析；其他平台需单独验证。

运行入口把 Ultralytics、Matplotlib 与 Torch 缓存放在项目 `.cache/`。本机 Ultralytics 设置也已指向项目 dataset/weights/results，并关闭同步和可选外部实验集成；这些本地设置不入 Git。新环境可在 `local_caches()` 后调用 `ultralytics.settings.update` 设置上述目录与 `sync=False`。

## 初始权重和真实预检

模型继续采用 [YOLO11n 检测模型](https://docs.ultralytics.com/models/yolo11/)，没有依据新模型发布改变首版方案。[官方 v8.4.0 权重](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt)保存到 `weights/yolo11n.pt`，大小5,613,764字节，GitHub digest 与本地 SHA-256 一致；详情见 `initial_weights.json`。

实际 CUDA 小矩阵运算通过；Ultralytics 扫描训练集510张全部有效。关闭增强加载 batch=8/640，COCO预训练模型保持 eval，在 inference_mode 和 AMP 下前向，输出 `[8,84,8400]` 有限；CUDA torchvision NMS 通过，梯度更新0。预训练头80类是初始状态，真正训练时才适配本任务3类。

前向峰值 allocated 约159.96 MiB，仅包括该检查的 PyTorch 张量分配，不能当作训练、显卡总占用或 batch=8 可训练的保证。反向传播、优化器、训练耗时、短跑保存权重和任务指标均未验证。

系统 Python 与项目 `.venv` 分别累计141项回归通过。解释器仍打印定位提示，但版本查询、导入、GPU运算与CLI实际正常完成，不单独把该提示当作执行失败。
