# 仅模型下载

2026-10-26，用户取消便携应用制作，改为只上传模型。当前默认仍为已验证的`goggles_v2_control`，后续微调没有取代它。

**[下载模型ZIP（5.71MB）](https://github.com/mocadesuwa/ML_ppe_detection/releases/download/goggles-v2-control-model/goggles_v2_control-model.zip)** · [发布页](https://github.com/mocadesuwa/ML_ppe_detection/releases/tag/goggles-v2-control-model)

包内只有`best.pt`、去除开发者路径的最小`experiment.json`推理契约、README及SHA256校验文件，没有EXE、Python/CUDA运行环境、原图或头像。模型支持CUDA加载；使用者仍需自行准备兼容的PyTorch环境，单独模型不等于可双击运行的应用。

| 项目 | 内容 |
| --- | --- |
| 模型 | torchvision MobileNetV3 Small，二分类末层 |
| 类别顺序 | 0=`not_wearing_goggles`；1=`wearing_goggles` |
| 输入 | 单人头部RGB头像，等比例填充224×224、ImageNet标准化 |
| 原训练环境 | PyTorch2.11.0+cu128、torchvision0.26.0、RTX3060 Laptop |
| 既往验证47张 | Accuracy91.49%、macro-F1 0.8494、佩戴Recall66.67% |
| 既往测试24张 | Accuracy79.17%、macro-F1 0.7052、佩戴Recall50% |
| ZIP大小 | 5,708,916字节 |
| ZIP SHA256 | `6dc8b0376e0bed242c1ed9a2d208f53b66c5c0b4cb46b98a5dedfc4c6caa3265` |
| 权重 SHA256 | `c66432ef67e71efb9341c7aceed7a2291e1f45cf4de313548ad3d9a03c6be8d7` |

只发布既有权重，没有重新训练、修改标签或重跑封存测试。包内最小契约保留完成训练、权重SHA、数据指纹、类别顺序及冻结/更新/重载的核验字段，附原训练记录SHA；不替代本地原始训练记录。

解压后保持`best.pt`和`experiment.json`相邻。在已有项目环境中使用现有预测入口：

```powershell
.venv/Scripts/python.exe -m src.predict_classification --weights <解压目录>/best.pt --source <单人头像或目录> --device 0 --name new_prediction
```

不输出位置框、不自动裁剪多人或全身图，分数未经校准；清楚覆盖眼部的防护镜为佩戴，普通眼镜、太阳镜及额头佩戴为未佩戴。完整效果和限制见[改进报告](goggles_improvement_report.md)。现有本机页面保留原样，使用说明仍见[本机页面](local_demo.md)。

本次已核验：原权重SHA、CUDA检查点加载、最小推理契约、GitHub附件状态/大小/服务端SHA，以及无需登录的公开下载、下载ZIP CRC与包内权重SHA。没有将发布核验称为新的独立准确率实验。
