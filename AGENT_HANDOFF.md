# 当前任务交接

更新日期：2026-10-26，北京时间。**当前网页操作台任务已完成，没有剩余实施。** 用户在模型发布后重新要求可操作内容，并明确选择“现有网页：启动项目后在浏览器操作”。这项新授权只完善网页，不恢复已取消的EXE/便携环境打包。

## 当前Git与交付

- 分支main，起点3cc9da0已同步origin/main。本手册与本轮网页/日志同次提交，准确提交号通过git log -1 -- AGENT_HANDOFF.md获取；最终交付前核对HEAD与origin/main一致及工作区干净。
- 启动入口仍是项目根目录start_demo.cmd，使用原.venv、默认GPU0和本地已验证模型。
- 当前默认goggles_v2_control，tail4仅作实验对照；没有重新选模或修改阈值。
- 已有模型Release及5.71MB附件保持不变：
  https://github.com/mocadesuwa/ML_ppe_detection/releases/tag/goggles-v2-control-model

## 本轮完成并验证

- 页面新增冻结数据划分、基线验证/历史测试指标、四组实验对照、逐轮曲线；可切换control/joint/tail2/tail4及验证macro-F1/训练与验证损失。
- 原头像选择/拖放预览、基线/微调模型切换、分类分数、重复提示与JSON下载保留；结果新增实际设备显示。
- 服务仅增加training.js与聚合training-summary.json静态路由；推理、同源限制、检查点契约和图片限制不变。
- JavaScript语法通过；完整234项回归先后24.056秒和22.900秒通过，终端输出同步build.log。
- 真实浏览器完成曲线/实验切换，宽窄窗口实际查看，中文显示正常；观察到窄窗口刻度过小后按容器宽度修正。
- 浏览器对一张已登记验证头像分别调用两个真实模型，均显示cuda:0；切换后旧结果清除。下载JSON核对candidate、cuda:0、val提示与两类分数和，通过。
- 浏览器无脚本error/warn；本地预览截图.cache/interactive-lab-overview.jpg不进入Git。最后标点空白修正已通过浏览器重新加载检查。
- 协议37份旧证据SHA不变；四份展示训练记录和已封存测试报告SHA一致。

## 文件及来源

- 代码/页面：src/serve_demo.py、tests/test_demo_server.py、web/dist/index.html、style.css、app.js、training.js、training-summary.json。
- 文档：docs/interactive_lab_plan.md、docs/local_demo.md、docs/framework_review.md、docs/change_logs/2026-10-26-02-interactive-model-lab.md、日志索引、README/CHANGELOG/AGENTS及本手册。
- 只复用此前缓存的网页草稿/聚合记录，并重新核对四份训练记录SHA和各轮macro-F1；没有恢复打包代码、虚拟环境或CPU应用。
- 所有源码改动均属于本次网页操作台范围，未安装依赖。真实图像、权重、缓存不进入Git。

## 边界、未验证与历史状态

- 234项回归中的HTTP使用替身推理；两个真实模型另由浏览器CUDA操作验证，未将替身当端到端结果。
- 使用既有验证头像核验功能，不是新图准确率实验。没有训练或重跑29/24封存测试；显示的79.17%与佩戴Recall50%仍是第二轮24张历史测试。
- 原生跨窗口文件拖放未自动化操作，拖放和文件选择共用后续处理。未验证公网、第二台机器或独立安装包，这些不属于已确认入口。
- 没有Qt应用、Qt offscreen或浏览器外渲染替代。预览服务按Sites本机指引在收尾停止，用户通过原启动器自行运行。
- 此前已取消的便携准备仍保存在.cache/cancelled-portable-20261026/，无需接续；模型仅Release发布仍有效。

## 额度与后续

本轮开始额度接口暂不可读，未猜测；完成验证后可靠读取为Plus、5小时used22%、周used11%，未达到Luna交接阈值。此为当时快照，不代表实时值；手册由当前Agent更新。

没有剩余实施任务。用户可启动页面使用；后续增加在线托管、训练控制或应用打包，需要新的明确授权，不自行展开。
