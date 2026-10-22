# 护目镜任务的头像准备接口

状态：通用头像数据准备接口，裁剪只执行显式头部框；另提供只读完整性检查和显式标签的分组划分计划。它们不寻找头部、不判断护目镜、不推断标签，也不生成模型专用数据格式。2026-10-22已按用户确认的二分类规则处理真实头像，具体版本和独立复核见[实际处理方案](data_processing_plan.md)；下文的中性例子仍只是格式示例。

## 输入清单

参照[示例JSON](../configs/head_crops.example.json)，复制后填写真实来源与已核查的框。示例路径和数值只是格式演示，不是真实数据清单。

```json
{
  "format_version": 1,
  "items": [
    {
      "id": "head_001",
      "source": "source/scene.png",
      "group": "source_group_001",
      "box_xyxy": [10, 20, 70, 100]
    }
  ]
}
```

| 字段 | 约定 |
| --- | --- |
| `id` | 输出头像编号，字母、数字、下划线或连字符；忽略大小写后不能重复，不接受Windows保留设备名 |
| `source` | 原图路径，相对该JSON所在目录解析，也可使用绝对路径 |
| `group` | 显式来源组；同图裁剪及内容完全相同的原图必须使用同一组。其他同人/同序列关系由实际核查补充 |
| `box_xyxy` | 四个整数像素坐标`[x1,y1,x2,y2]`；左上包含、右下不包含，输出宽高为`x2-x1`、`y2-y1` |
| `label` | 可选，保留调用者给出的非空字符串；不转换类别，不核实佩戴含义，不填写时不生成标签 |

框必须严格位于原图范围内且宽高为正。工具不自动扩边、截断、旋转或缩放；当前课程规则见实际处理方案，本通用接口不内置业务类别或裁剪比例。非默认EXIF朝向的输入会被拒绝，应先核对像素朝向与框坐标。仅接受JPG/JPEG、PNG、BMP和静态WebP，拒绝动画、损坏图片和重复JSON字段。

## 执行与输出

以下为通用示例调用；真实处理使用独立的最终框清单，不直接运行示例路径：

```powershell
python -X utf8 -m src.prepare_head_crops --manifest configs/head_crops.example.json --output dataset/goggles/heads_v1 --dry-run
python -X utf8 -m src.prepare_head_crops --manifest configs/head_crops.example.json --output dataset/goggles/heads_v1
```

`--dry-run`会读取原图尺寸与哈希、检查全部框并打印计划，不创建输出。它与不读取数据内容的训练计划dry-run用途不同。

正式生成必须使用新输出目录，且不得位于任何原图所在目录中。先完成全部预检，再在临时目录逐张生成RGB PNG；清单和原图哈希前后检查通过后，才将整个目录发布到目标位置。失败不发布不完整头像目录，不覆盖既有输出。

Windows目录发布遇到访问/占用错误码5、32、33时最多尝试6次，总等待最多1.55秒，每次重新检查目标不存在。其他错误或持续失败直接报错，不改用逐文件复制，也不覆盖已有目录。

输出结构：

```text
heads_v1/
  images/head_001.png
  crops_manifest.json
```

`crops_manifest.json`记录输入清单哈希、原图路径/哈希/尺寸、每个头像的编号、来源组、原图框、输出尺寸、文件及哈希，还有调用者提供的可选标签。`status=complete`只表示本次裁剪完整，`visual_review_completed=false`明确尚未完成人工质量核查；它不是训练数据就绪记录。

输出后仍需实际确认眼部和护目镜区域完整、标签含义一致。分类路线需头像级标签；检测路线还需明确框语义并转换相应标注，本工具不替代这些步骤。

## 生成后只读检查

```powershell
python -X utf8 -m src.check_head_crops --crops dataset/goggles/heads_v1
```

程序重新读取原始清单、原图和头像，检查：

- 清单字段、记录数量、编号/标签/来源组/裁剪框与原始清单完全一致。
- 输入清单、原图、头像哈希一致，目录没有缺失、额外文件或符号链接。
- 头像确为静态RGB PNG，尺寸正确，像素与原图指定区域逐一相等；即使替换图片后更新了输出哈希也会拒绝错误裁剪。
- 检查前后内容不变；返回绑定清单、来源、标签和实际头像内容的`crop_fingerprint`。

结果JSON输出到标准输出，工具不写文件。原始清单和原图必须仍在记录的位置；移动或修改它们需要重新建立数据版本。`integrity_passed`只代表文件与给定裁剪规则一致，返回标签计数、未标注数量，始终保留`visual_review_completed=false`、`training_ready=false`，不能代替人工判断裁剪是否合理。

## 分组划分计划

标签及来源组实际明确后，可单独生成划分清单。下例的`class_a/class_b`是中性格式示例，必须替换为已确认且与每项`label`精确一致的名称，不是课程类别要求。上面的两张无标签演示图不能运行此划分。

```powershell
python -X utf8 -m src.split_head_crops --crops dataset/goggles/heads_v1 --classes class_a class_b --ratios 0.7 0.2 0.1 --seed 42 --output dataset/goggles/splits_v1 --dry-run
python -X utf8 -m src.split_head_crops --crops dataset/goggles/heads_v1 --classes class_a class_b --ratios 0.7 0.2 0.1 --seed 42 --output dataset/goggles/splits_v1
```

两个命令都会执行完整性检查，并验证输出目录为新目录、位于头像与原图目录之外；`--dry-run`只打印计划，不创建文件。正式运行在重新核对输入及计划后发布`splits_v1/splits_manifest.json`，不复制或修改图片、标签。

规则如下：

- 每张头像必须有显式标签，不能缺失或出现未声明类别；每个声明类别必须实际出现。
- 将显式来源组、内容完全相同的原图、解码像素和尺寸完全相同的头像进行传递合并。同图的不同头像、重复头像及其关联来源组不可跨集合；完全相同头像有不同标签时拒绝划分。
- 复用已有分组算法，固定输入顺序、类别顺序、比例和种子可复现；每类至少需要3个独立组，并实际覆盖train/val/test，无法满足时报错，不拆组凑数。
- 比例是目标而非精确保证。清单保存实际样本/类别计数、合并后的组编号、原始来源组、输入指纹及参数，便于检查偏差。

该清单是`status=planned`的划分计划，不是可直接训练的分类目录或YOLO标注；保留`training_ready=false`，后续仍需人工质量核查和最终任务格式适配。程序不识别人脸身份、拍摄序列或近重复关系，必须先人工把已知关系填入`group`，来源不明时记录限制。裁剪或标签改变后指纹也会变化，应建立新版本，不能继续沿用旧划分。

## 无模型合成演示

```powershell
python -X utf8 scripts/demo_head_crops.py --output .cache/head_crop_demo
```

使用新的目录名运行，生成一张示意图、显式框清单、两张头像和来源记录。不访问真实数据、不加载模型、不安装依赖。演示没有佩戴标签，不能作为训练样本或识别效果证据。Windows捕获命令行JSON时使用`-X utf8`可统一输出编码，清单和结果文件本身使用UTF-8。
