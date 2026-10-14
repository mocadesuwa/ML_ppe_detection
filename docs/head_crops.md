# 护目镜任务的头像准备接口

状态：头像数据准备功能，可在未确定分类或检测路线前使用。本工具只按显式头部框裁剪，不寻找头部、不判断护目镜、不推断标签，也不自动划分训练集。

## 输入清单

参照[示例JSON](../configs/head_crops.example.json)，复制后填写真实来源与已核查的框。示例路径和数值只是格式演示，本轮未读取或生成真实数据。

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

框必须严格位于原图范围内且宽高为正。工具不自动扩边、截断、旋转或缩放；当前课程的头部保留范围及佩戴类别仍待明确。非默认EXIF朝向的输入会被拒绝，应先核对像素朝向与框坐标。仅接受JPG/JPEG、PNG、BMP和静态WebP，拒绝动画、损坏图片和重复JSON字段。

## 执行与输出

以下为待实际数据工作推进后的入口，当前仅用合成样本验证：

```powershell
python -X utf8 -m src.prepare_head_crops --manifest configs/head_crops.example.json --output dataset/goggles/heads_v1 --dry-run
python -X utf8 -m src.prepare_head_crops --manifest configs/head_crops.example.json --output dataset/goggles/heads_v1
```

`--dry-run`会读取原图尺寸与哈希、检查全部框并打印计划，不创建输出。它与不读取数据内容的训练计划dry-run用途不同。

正式生成必须使用新输出目录，且不得位于任何原图所在目录中。先完成全部预检，再在临时目录逐张生成RGB PNG；清单和原图哈希前后检查通过后，才将整个目录发布到目标位置。失败不发布不完整头像目录，不覆盖既有输出。

输出结构：

```text
heads_v1/
  images/head_001.png
  crops_manifest.json
```

`crops_manifest.json`记录输入清单哈希、原图路径/哈希/尺寸、每个头像的编号、来源组、原图框、输出尺寸、文件及哈希，还有调用者提供的可选标签。`status=complete`只表示本次裁剪完整，`visual_review_completed=false`明确尚未完成人工质量核查；它不是训练数据就绪记录。

输出后仍需实际确认眼部和护目镜区域完整、标签含义一致。分类路线需头像级标签；检测路线还需明确框语义并转换相应标注，本工具不替代这些步骤。

## 无模型合成演示

```powershell
python -X utf8 scripts/demo_head_crops.py --output .cache/head_crop_demo
```

使用新的目录名运行，生成一张示意图、显式框清单、两张头像和来源记录。不访问真实数据、不加载模型、不安装依赖。演示没有佩戴标签，不能作为训练样本或识别效果证据。Windows捕获命令行JSON时使用`-X utf8`可统一输出编码，清单和结果文件本身使用UTF-8。
