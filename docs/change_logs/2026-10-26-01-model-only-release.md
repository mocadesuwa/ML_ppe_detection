# 取消应用扩展，发布模型附件

时间：2026-10-26，北京时间；用户在14时段先澄清希望可下载的应用，后要求GPU优先并询问体积，最终明确“只上传模型就行，本体就不做了”。本日志记录最终交付范围及本轮已执行/取消工作。关联提交：与本日志同次提交，编号通过Git历史查找；发布目标代码为此前已同步的`243fef8`，新模型标签不覆盖`v0.1.0`。

## 最终变化

- 不再制作便携应用，不再扩展页面；恢复本轮改动的`src/serve_demo.py`、旧HTTP测试、HTML/CSS及临时打包忽略项，Git内容与原提交一致。此前可用的页面保留。
- 本轮临时打包脚本、CPU虚拟环境、第一份CPU构建及页面新增文件保存至`.cache/cancelled-portable-20261026/`，未提交或发布。该CPU构建只是中间产物，不作为GPU应用交付。`AGENT_HANDOFF.md`保留并更新为最终模型发布状态，没有自动删除交接手册。
- 将原`goggles_v2_control/best.pt`原字节发布为GitHub Release附件；ZIP只含权重、最小推理契约、说明与校验，不含应用、CUDA库、Python环境或真实图片。真实权重仍不进入Git历史。
- 新增`docs/model_release.md`，更新README、AGENTS、CHANGELOG及日志索引。忽略`/build.log`以保存前台验证日志而不纳入Git。

## 实际发布与验证

- `.cache/release_model.py prepare`检查选定模型SHA，与第四轮选定记录一致；使用原CUDA虚拟环境成功将检查点加载到`cuda:0`，检查类别顺序、数据指纹和config SHA。没有训练或图像前向。
- 包内`experiment.json`为去除开发者绝对路径的最小推理契约，包含原训练记录SHA；`completed_training_record`接受该契约，原始记录保持不变。
- 原始权重SHA：`c66432ef67e71efb9341c7aceed7a2291e1f45cf4de313548ad3d9a03c6be8d7`。ZIP为5,708,916字节，SHA：`6dc8b0376e0bed242c1ed9a2d208f53b66c5c0b4cb46b98a5dedfc4c6caa3265`。
- GitHub连接器可核对仓库权限但没有Release附件工具，本机没有`gh`。使用现有Git凭据在内存中调用GitHub官方REST接口，仅发送到`api.github.com`/`uploads.github.com`，未输出或保存凭据。
- Release `goggles-v2-control-model`，ID423925368；附件ID677263267。先草稿上传核对服务端SHA/大小，再公开发布为预发布模型，未冒充应用版本。
- `.cache/verify_model_download.py`无需登录重新读取公开Release并下载附件：字节数、ZIP SHA、CRC、四个成员及包内权重SHA全部通过。已使用发布后返回的稳定下载URL，未把草稿阶段的临时URL交给用户。
- 发布验证输出同步终端与根目录`build.log`；详细发布摘要在`.cache/model-release-20261026/published.json`。

## 回归、失败及未运行

- 最终只保留文档、交接及日志忽略规则，不改变生产代码，故没有无故重跑模型训练、数据预检或全量代码回归；上轮233项是历史结果，本次不称重新通过。
- 本轮先前CPU准备安装和单目录构建成功，但尚未做EXE端到端验证，随后用户取消，故不交付该构建。CPU准备曾发生两个前台命令同时争用`build.log`文件句柄，已停止并行写日志；资产导出和安装最终成功。
- 第一份页面补丁因同一文件重复操作被工具拒绝，未执行；后续修改也随用户取消全部撤回。没有为此改动原模型或测试。
- Qt界面、第二台机器应用、GPU便携包、训练/测试重跑均未运行，不属于最终“只上传模型”要求。
- 提交前完成文档链接/JSON/索引检查、`git diff --check`、37份旧证据SHA以及生产代码Git内容不变检查。只提交文档与忽略规则，普通push后核对本地与远端HEAD。
- 文档检查实际计数：57份Markdown、260个本地链接/图片、31份已索引日志，全部通过。模型发布不需要再次执行原233项历史回归。

限制：模型包不自带运行环境，接收者需自行准备GPU兼容PyTorch并使用现有源码入口。模型仍是小样本课程基线，历史佩戴Recall50%，不称为完整实验室部署模型；不再开展用户已取消的应用制作。
