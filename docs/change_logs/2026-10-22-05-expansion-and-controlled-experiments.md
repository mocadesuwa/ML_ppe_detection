# 第05轮：新增来源、v3数据与有界对照实验

- 记录时间：2026-10-22 19:30（北京时间，UTC+08:00）。提交编号通过本日志Git历史查找。
- 原因：用户进一步授权“扩充数据集，或者选取其他方式，尝试提高准确率”。更新前一轮只使用冻结小样本的工作边界；保留原v2数据、旧模型和旧29张测试结果，不通过反复测试调参。

## 数据获取与实际视觉复核

从已核验的SH17作者Kaggle固定版本1下载140张原图与原VOC，原图224455039字节，全部成功，未下载全量数据。作者版本/仓库固定到原来源记录的revision，声明CC BY-NC-SA4.0。镜像只用于候选位置，逐图与作者原VOC框匹配，使用作者原始像素和原标注，不采用镜像像素或镜像宽松许可。保留原件、CRC单成员ZIP检查、图像尺寸/朝向及图像/标注SHA；命令`python -X utf8 -u .cache/expand_sh17_20261022.py`退出0，完整输出`.cache/sh17-expansion-20261022.txt`。

实际查看9张原场景/头像拼图，覆盖全部140项；另打开疑难头像。保留110张：佩戴18、未佩戴92；排除30张眼部不可判断、模糊、不完整、多头或类型不确定样本。普通眼镜、太阳镜、额头佩戴仍为未佩戴，不依据上游glasses直接标正例。标签由Codex视觉复核，未进行独立第二标注者复核。

旧SH17原80张所在73个数字来源桶全部排除，新候选来自其他127个桶；相近系列与保守视觉场景家族合组。对旧191张加新110张的301张头像，筛查涉及新增项的跨组dHash≤3或pHash≤6，候选0，证据`.cache/goggles-v3-similarity.json`；哈希筛查和数字桶不等于人物/摄影者绝对独立。

`python -X utf8 -u .cache/prepare_expanded_goggles.py`退出0：新建原生RGB头像、逐项审查、精确像素检查与ImageFolder冻结导出。累计301张有效头像，29张旧测试封存不进入本轮；新v3共272张，train201（157/44）、val47（38/9）、新test24（18/6）。训练来源组从12增至72，佩戴来源组从3增至8。保留旧train/val用途，新增整组划分先于任何新模型前向；标签、框、原件和旧划分未覆盖。

数据指纹`5859b079a61f07f9436f5a7c75383d9c8854fe73fb98c0211cb4045643ae550c`。新增来源、复核、头像框、审核和固定成员保存在`docs/data_sources/*v3*.json`及`sh17_expansion_140_20261022.json`；实际图片继续在Git忽略的`dataset/goggles/`。

## 代码、配置与选择规则

- `src/train_classification.py`支持显式`sampling=source_group_balanced`及`class_weighting=none`；来源组内每张抽样权重为组大小倒数，按完整训练ID核对，有放回每轮201次。默认仍沿用原逐图shuffle和训练频数类别权重。骨干冻结、加权loss累计和最佳权重重载检查保留。
- `src/evaluate_classification.py`新增显式`--benchmark`，仅用于完成的旧模型在新独立测试集评价。要求新test与旧所有train/val/test在ID、来源组、PNG/像素、原图字节上无已知交叉；保留旧检查点真实配置和数据指纹，使用新数据指纹命名的一次性记录，不覆盖或绕过旧最终测试。
- 三个有界配置control/groups/uniform，均从原ImageNet权重独立初始化，最多30轮、patience8；分别为原方法、来源组均衡且类别加权、来源组均衡且不加类别权重。短跑配置2轮，预期26次更新；不扩大模型或增加依赖。
- `docs/goggles_improvement_protocol.json`在任何新梯度训练之前写入三份配置SHA、数据指纹、旧证据SHA、选模与评价规则：最高验证macro-F1、再按该轮验证Accuracy、仍同分按候选顺序。测试不调阈值或选模型。
- 同步AGENTS、需求、框架、README、来源索引和`docs/goggles_improvement_plan.md`，明确模型v2使用数据v3及第一轮结果的历史性质。

## 实际验证与前序回归

`python -X utf8 tests/run_regression.py`退出0：212项，22.908秒，全部通过。新增9项覆盖来源组抽样质量、未知/重复ID、无类别加权配置、同数据不能绕过旧测试、新独立marker、旧test/像素/原图重叠拒绝及smoke拒绝。包含上一轮203项及更早回归，完整输出`.cache/regression-2026-10-22-expansion-01.txt`。

只读调用`independent_benchmark(v3,v2)`退出0，排除旧全部191张，新test24张，已知交叉0，参考数据指纹与旧模型记录一致；没有模型前向。证据`.cache/goggles-v3-benchmark-independence.json`。来源审核与导出像素/哈希核查均通过，原v2文件未修改。

control、groups、uniform与smoke四份配置的真实CUDA无梯度预检均退出0，均为`ready_before_training`、更新0、参数/缓冲区不变，原始两类16张前向通过，峰值已分配54.7158MiB；批次结果`.cache/preflight-goggles-v2-all.txt`与四份独立报告保留。本日志记录的是新训练前修改，不把待执行拟合或评价写成已通过。

文档/入口检查退出0：40份Markdown的UTF-8/围栏、140个本地链接及图片、24份日志索引和docs内JSON解析通过；两个修改的CLI帮助子进程及src/tests编译通过，证据`.cache/goggles-v3-doc-checks.json`。`git diff --check`退出0，仅提示既有Git换行策略。

## 限制、同步与下一步

新test仅6个佩戴样本、4个佩戴来源组；样本仍为公开工业/素材域，未完成真实实验室验证和独立复标。下一步先短跑验证26次真实更新、冻结、头更新和重载，再执行三组对照；选定后在同一新test各评价旧模型和选中模型一次，未选模型不接触新test。

未安装依赖。按既有授权提交并普通push代码、配置、元数据和聚合结果，不上传原图、头像、权重或完整逐图输出。日志不预填尚未执行的训练、提升幅度或push结果。
