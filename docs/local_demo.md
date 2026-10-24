# 本机拖拽识别页

在项目目录双击 **`start_demo.cmd`**，或在PowerShell中执行：

```powershell
.\start_demo.cmd
```

启动后自动打开 `http://127.0.0.1:8765`。保持服务窗口运行，关闭时按`Ctrl+C`。端口被占用时先关闭之前启动的服务；也可按下方命令改端口。不要直接双击`web/dist/index.html`，页面需要Python推理接口。

## 操作

1. 将一张**单人头部裁剪头像**拖入左侧，或点“选择图片”。保留完整眼部；多人、全身图片需先自行裁剪。
2. 默认使用当前基线。点“开始识别”，右侧显示佩戴/未佩戴判断、两类分数、图像尺寸及本次推理耗时。首次识别包含模型加载，会稍慢。
3. 可以更换图片、切换微调实验模型，并下载本次判断JSON。微调模型只供比较，本轮没有优于基线，不默认替换。

支持静态JPG、PNG、BMP、WebP，单张不超过8MiB，至少16×16，总像素不超过1600万。动图、损坏文件和超限图片会提示错误。分数未经概率校准，不能视为真实正确率。输入不符合头像要求时仍可能输出二分类结果，本模型没有自动判断“非头像/无法判断”的第三类。

佩戴规则为防护镜正确覆盖眼部；普通眼镜、太阳镜或额头佩戴归为未佩戴。看不清的图片不适合评价模型效果。

## 模型与数据

- 当前默认：`results/classification/goggles_v2_control/best.pt`。
- 可选实验：`results/classification/goggles_v4_tail4/best.pt`，验证macro-F1与基线同为0.8494；无新的独立测试。详细训练比较见[第四轮报告](goggles_finetuning_report.md)。
- 输入图片在本机内存中读取，不写入上传目录、不向外部服务发送、不自动加入训练集。只有主动点击下载时，浏览器才保存判断JSON。
- 与当前模型登记的train/val/test头像在文件字节或RGB像素上完全相同时，页面提示所属集合。这里只读取清单哈希，不主动预测任何测试集。未命中不代表已经证明是独立来源；也不能识别“同一人换了一张照片”。
- 页面可接收模型未见过的新图片，但模型训练的泛化能力仍受小样本限制。建议使用清晰、独立拍摄的新头像逐张观察。

本地虚拟环境、真实数据和权重没有上传Git。另一台电脑仅克隆代码后不能直接识别，需自行准备相同依赖及本地有效检查点；程序不会自动下载权重、自动训练或改变数据。

## 命令与接口

无需新依赖。前端是原生HTML/CSS/JavaScript，后端用Python标准库HTTP服务、现有Pillow和PyTorch。手动启动示例：

```powershell
.venv/Scripts/python.exe -X utf8 -m src.serve_demo --weights results/classification/goggles_v2_control/best.pt --candidate results/classification/goggles_v4_tail4/best.pt --device 0 --port 8765 --open-browser
```

只使用基线时去掉`--candidate`，使用CPU时改为`--device cpu`。服务仅监听`127.0.0.1`，只允许本机同源请求，不提供公网或局域网访问；静态路由只开放三个前端资源，不开放数据、权重及项目目录。

| 接口 | 请求 | 返回 |
| --- | --- | --- |
| `GET /api/models` | 无 | 可选模型ID、名称、说明及上传上限 |
| `POST /api/predict` | 原始图片二进制body，`X-Model-Id: baseline`或`candidate` | `label`、`probabilities`、尺寸、耗时、图像/权重SHA及`known_splits` |

请求采用图片原始字节，不使用multipart。类别为`not_wearing_goggles`/`wearing_goggles`，没有位置框。服务启动时核对完整训练记录和权重SHA，首次推理核对检查点契约；运行中模型/记录变化将要求重启。

验证包括真实浏览器的文件选择、双模型识别、超限提示和JSON下载，以及8次真实验证头像接口调用。原生跨窗口拖放未自动化操作，拖放代码与文件选择共用后续处理。完整回归233项通过；浏览器检查使用验证头像，只验证功能，不作为新图准确率证据。
