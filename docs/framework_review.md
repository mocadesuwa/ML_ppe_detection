# 护目镜项目代码框架

更新日期：2026-10-24。用户已确认静态头部头像二分类，仅输出佩戴状态、不输出框，详见[需求](goggles_requirements.md)。当前仍使用272张v3数据选出的control；第四轮微调未优于基线，新增本机交互页面，见[最新报告](goggles_finetuning_report.md)。第二轮同一新保留集结果见[改进报告](goggles_improvement_report.md)，其余数据/模型保留历史。

## 护目镜主流程

原图和原标注 → 显式复核与头部框 → `prepare_head_crops` → `check_head_crops` → `split_head_crops` → `classification_data` → 分类预检 → 分类训练、评价与预测。

当前数据统一在`dataset/goggles/`，当前模型和评价在`results/classification*`；具体目录见[数据目录说明](../dataset/README.md)，当前与历史配置见[配置索引](../configs/README.md)。

| 文件 | 职责 |
| --- | --- |
| `src/common.py` | 路径、配置、哈希、数据指纹、质量记录、实验目录及原子发布的有界Windows重试 |
| `src/import_data.py` | 显式本地ZIP导入、安全解压及来源记录，不联网 |
| `src/source_review.py` | 显式类别的VOC源数据只读核查：尺寸、对象、两种坐标约定、重复冲突与来源指纹 |
| `src/prepare_head_crops.py` | 显式头部框预检、RGB头像裁剪、来源组约束及输入/输出哈希记录 |
| `src/check_head_crops.py` | 只读核对原始清单、原图区域与头像像素、目录完整性及头像指纹 |
| `src/split_head_crops.py` | 显式标签、同源/重复头像合组、可复现划分计划，不生成模型格式 |
| `src/classification_data.py` | 显式审查/来源域绑定、ImageFolder导出、冻结目录/哈希/标签/同源跨集合核查 |
| `src/train_classification.py` | 当前分类训练计划；默认/`--dry-run`不训练，`--start`额外检查匹配预检后才构建训练模型/优化器 |
| `src/classification_model.py` | 分类检查点契约、完成运行/最佳权重绑定、两类指标与概率输出检查 |
| `src/evaluate_classification.py` | 分类验证/最终测试、逐图分数与错误记录，最终测试仅允许完成的基线且防止重复尝试 |
| `src/predict_classification.py` | 已裁剪静态头像的类别预测，逐图JSONL与输入/权重完整性检查 |
| `src/serve_demo.py` | 本机图片解码、已完成模型校验、内存推理及只开放前端资源的HTTP接口 |
| `web/dist/training.js`、`training-summary.json` | 读取已存聚合训练记录，切换实验/曲线与展示数据划分，不触发训练或测试 |
| `web/dist/` | 无构建依赖的本机拖放识别页面与结果下载 |
| `scripts/launch_frontend.py`、`scripts/create_frontend_shortcut.ps1` | 图形启动器、所属服务生命周期和本机快捷入口；原`start_demo.cmd`保留备用 |
| `src/environment.py` | 查询现有Python、依赖和GPU环境 |
| `scripts/find_related_images.py` | 相似图片候选，不自动判断人物或来源组 |
| `scripts/preflight_classification.py` | 当前分类数据加载、变换、官方初始模型与CUDA eval前向；不构建优化器、不训练 |
| `scripts/demo_head_crops.py` | 纯合成头像流程演示，不生成业务标签或训练数据 |

## 通用检测基础

下列工具接受调用者显式提供的类别和路径，不包含专用业务类别。分组算法和静态图片预检也供头像分类流程复用，因此保留在通用框架中。

| 文件 | 职责 |
| --- | --- |
| `src/prepare_dataset.py` | VOC到YOLO转换、去重、来源分组与固定划分；头像划分复用其中的分组算法 |
| `src/check_dataset.py` | 检测标签、清单和划分校验，生成标注抽查拼图 |
| `src/review_dataset.py` | 记录实际人工核查并绑定数据指纹 |
| `src/train.py` | 显式检测训练计划，`--dry-run`不读取真实数据内容或加载模型 |
| `src/evaluate.py` | 显式检测数据/权重、类别一致性、训练来源及最终测试约束 |
| `src/predict.py` | 静态图片预检及检测结果；分类预测复用其中的静态输入检查 |
| `scripts/preflight_training.py` | 显式检测配置的无梯度预检 |

当前护目镜使用独立分类入口，输出佩戴状态和两类分数。旧专用业务分支及本地数据/检查输出已清理；通用算法、校验和测试继续保留。完整清理范围及保护核查见[本轮日志](change_logs/2026-10-22-07-remove-mask-data.md)。

第一轮按[分类训练流程](classification_training_workflow.md)完成短跑、基线与冻结测试。分类入口已补齐，损失日志按类别权重分母累计；该阶段203项回归与真实运行证据保留历史，不把回归通过视为模型效果。

用户进一步授权提高准确率，本轮完成[数据扩充与有界对照](goggles_improvement_plan.md)：训练可显式选择来源组均衡抽样或不加类别权重，默认仍是原方案；评价新增`--benchmark`仅允许旧模型在与旧全部集合无已知交叉的新测试上比较，保留原模型指纹和旧测试记录。212项累计回归通过；三组均完成，按验证选择control，当前结果见[摘要](goggles_improvement_status.json)，不把回归或验证指标代替新测试比较。

2026-10-23按[第三轮方案](goggles_joint_sampling_plan.md)新增`class_source_group_balanced`，让每类抽样总质量相等、类内来源组均衡，且强制不再加类别损失权重；旧两种策略保持不变。新增3项回归，215项通过；本轮只安排一个完整候选，选择不通过时继续保留control，不重跑已封存测试。

第三轮已完成短跑26次、完整训练130次更新，候选验证macro-F1为0.8321、佩戴Recall55.56%，未超过control，继续保留原模型。100张新来源核查后未并入，也没有建立新的独立测试；完整证据及具体数据缺口见[本轮报告](goggles_joint_sampling_report.md)。

第四轮支持从已完成任务检查点进行尾部微调：配置显式解冻模块数及两组学习率，预检绑定初始化记录，训练核对冻结前部、归一化缓冲区及允许更新部分。旧头部训练和检查点继续可用，微调不会把整个特征提取部分谎报为冻结；具体设计见[第四轮方案](goggles_finetuning_plan.md)。

第四轮短跑及两个完整候选已完成，均未达到替换条件，control继续作为默认。交互页面复用分类检查点契约和既有变换，不另建模型/训练分支；只监听127.0.0.1，图片只在内存中处理。字节/像素匹配仅提示登记集合，不保证来源独立。233项回归、真实模型接口和浏览器上传/下载核验已完成，启动说明见[本机页面](local_demo.md)。

2026-10-27页面以头像上传和识别结果为主体，训练曲线、实验对照、数据与规则由小按钮按需展开；`training.js`只管理辅助面板与历史摘要展示，不改变`app.js`的识别流程。当前234项回归与浏览器真实CUDA识别通过，见[网页方案](interactive_lab_plan.md)。

同日收尾只修正`app.js`的识别错误反馈：连接中断与服务返回的业务错误分开提示，恢复后原图可重试，收到响应再恢复顶部连接状态。模型、接口、数据和训练流程保持不变，见[收尾日志](change_logs/2026-10-27-02-recognition-feedback-closeout.md)。

## 当前分类路径

2026-10-28本轮只完善前端，并增加Tkinter图形启动器。启动器以现有虚拟环境调用`src.serve_demo`，保持GPU0、8765端口和现有模型选择；命令行后台运行，状态/停止由真实图形窗口提供。`src/`中的接口、分类、训练和数据代码均未修改。新窗口和新页面实际操作尚待人工验证，自动检查边界见[前端方案](frontend_launcher_plan.md)。

真实逐项复核和框清单见[数据处理方案](data_processing_plan.md)。最终头像由既有裁剪/像素检查生成，固定整组划分后，再按来源覆盖调整完整组；`classification_data`要求全部头像已审查、类别字母顺序一致、每类覆盖三个集合，审核附件哈希与头像指纹匹配；显式组、同源字节和相同像素不允许跨集合。提供来源域规则时，还要求每个域的两类均进入训练。

导出时复制头像原始PNG字节，在临时目录核查后发布新目录；检查器拒绝未知/重复路径、缺失/额外文件或目录、符号链接、内容/标签/来源变化和陈旧审核。冻结指纹同时绑定头像、划分计划和质量记录。

预检使用ImageFolder加载两类真实训练样本，等比例缩放填充224并做ImageNet标准化；训练专用的轻微增强先检查内存输出。官方MobileNetV3 Small初始权重换上未训练的两类头，冻结特征，`eval`/`inference_mode`前向并核对参数及缓冲区不变，预检不在val/test上做模型前向。之后真实训练才加载训练增强，验证和最终测试使用无增强输入。

分类训练入口默认只输出核查后的计划，显式`--start`才核对预检并构建模型/优化器，测试不进入训练循环。第一轮18/99次更新见[原训练状态](training_status.json)。第二轮短跑26次更新通过后，control完成11轮143次更新、groups和uniform各14轮182次更新；均通过骨干冻结、头更新和最佳权重重载。按预先固定的验证规则选定control后，才评价新保留集，实际指标见[改进状态](goggles_improvement_status.json)。

## 源数据核查

`python -m src.source_review`必须提供`--raw`、`--classes`和`--output`。类别名称与源XML精确对应；未知对象记录错误，不能静默跳过。两种坐标约定分别检查，不自动选定哪种为真值。

输出包含类别表、全部对象、重复冲突、来源哈希与前后不变检查；拒绝覆盖已有输出或写入原始数据内。报告明确`visual_review_completed=false`，程序核查不等同于人工看过图片。本工具不修改、修补或裁剪原始数据。

## 通用检测结果格式

本节仅描述`src.predict`的扩展接口。当前头像分类由`src.predict_classification`输出类别与两类分数，输入裁剪规则见[需求](goggles_requirements.md)。

- 支持JPG/JPEG、PNG、BMP和静态WebP。空文件、无法校验的图片、动画及单文件视频在模型加载前报错。
- 目录只筛选直接包含的支持图片，逐张交给后端；没有图片时报错，不一次加载整个数据集。
- `predictions.jsonl`每张包含`source`、从0开始的`image_index`、`image_size`和`detections`；检测项包含类别、置信度与像素框。空检测不能直接表示确定未佩戴。
- `run.json`使用`format_version=2`、`image_count`，绑定输入/权重哈希与实际参数，记录`running/complete/failed`。
- 保留重复/遗漏、框/类别/置信度及内容变化检查；结果先写`.part`，通过完整检查后才发布，失败或中断保留证据。

通用检测入口不能认定输入已经正确裁成头部头像。

## 头像准备

新增[头像清单接口](head_crops.md)，用显式像素框裁剪，与分类/检测选型无关。逐项校验编号、来源组、静态图片、朝向和框；同一原图或相同内容的所有裁剪必须同组。保留可选标签，不推断佩戴状态。

全部预检通过后逐张生成RGB PNG，在临时目录完成记录与前后哈希检查，再发布新输出目录。原件、已有输出均不覆盖；裁剪完成不代表视觉审核或模型完成。本轮真实头像已生成并另作逐图视觉复核，通用工具本身仍返回视觉审核未完成；分类导出使用单独、显式、绑定指纹的实际复核记录。

生成后可只读核查哈希、尺寸、RGB PNG格式及逐像素内容，拒绝遗漏或额外文件；指纹绑定裁剪规则、来源和标签。来源原图和原始清单须仍可访问，核查不证明框符合实际头部规则。

标签完整且显式类别一致时可生成分组计划。显式来源组、相同原图字节和相同头像像素传递合并，复用已有`split_groups`算法；不拆组凑比例，每类须在三个集合中出现。计划保存实际计数和输入指纹，不复制图片，不选择分类/检测格式，仍需核实人物、序列、近重复关系与人工质量。

本机回归暴露目录重命名和JSON原子替换的间歇性WinError 5。共用`publish_path`只对Windows错误码5/32/33进行有界重试，最多6次，其他或持续错误仍失败；目录目标每次检查不存在，JSON替换失败保留先前完整记录，不用逐文件复制替代发布。

## 验证与下一步

回归使用临时合成样本和假模型，中性类别名仅为测试数据。按既定顺序检查坐标、分组、解压、清单/指纹、标签、配置、评价、预测和来源核查；代码变化后运行`python tests/run_regression.py`。

数据准备阶段189项回归与零更新预检保留在[前阶段日志](change_logs/2026-10-22-02-data-preparation-pretraining.md)和[历史快照](pretraining_status.json)。第一轮增至203项及模型结果见[原报告](goggles_training_report.md)；第二轮增至212项及同一新保留集比较见[改进报告](goggles_improvement_report.md)。两轮均未安装依赖，只同步代码、配置、聚合结果与记录，原图和权重仅留本地。独立复标、实际实验室域验证和课程合格门槛仍未完成。
