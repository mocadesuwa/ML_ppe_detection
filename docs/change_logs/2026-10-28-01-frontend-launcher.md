# 图形启动器与识别前端

- 日期：2026-10-28，北京时间；23:11收尾记录。
- 原因：用户希望摆脱CMD启动方式，为已有项目提供可用前端；训练和数据任务明确排除。
- 开始状态：`main`，HEAD为`ff6c9fa`，工作区干净。当前实现、文档和本日志同次提交，准确提交号通过本文件Git历史查询。

## 具体修改

- 新增`scripts/launch_frontend.py`：Tkinter图形启动器，以现有虚拟环境调用原`src.serve_demo`。自动打开本机页面，提供状态、打开页面、停止/重启、日志和单实例管理。只清理自己创建的服务进程，仍使用GPU0和8765端口。
- 新增`scripts/create_frontend_shortcut.ps1`，并在当前项目根目录实际生成`护目镜识别.lnk`。快捷方式包含本机路径，加入`.gitignore`；可用脚本重建，没有安装运行环境或修改系统策略。
- 修改`web/dist/index.html`、`style.css`、`app.js`：Neutral Modern浅灰/钴蓝中文工作台，识别模型选择位于上传侧，图片完整预览、清空、尺寸/大小、读取状态、重连与识别反馈。读取图片与推理使用独立序号，避免切换模型使尚在解码的图片失去状态；更换输入/模型会清理旧判断与下载。
- 辅助资料保持原有内容和小按钮交互；`training.js`和`training-summary.json`未改。未改`src/`、配置、数据集、权重、实验记录或原CMD入口，没有新增素材、依赖或安装包。
- 新增`tests/test_frontend_launcher.py`，覆盖真实子进程终止、中文日志与退出事件、Windows原生单实例句柄。没有将这些检查称为GUI验证。
- 同步README、本机说明、框架说明、`web/README.md`、CHANGELOG、[本轮方案](../frontend_launcher_plan.md)、日志索引及根交接手册。

## 验证与回归

命令在前台执行，终端输出同步保存于根目录`build.log`（本地忽略文件）。最终代码修改后运行：

| 检查 | 实际结果 |
| --- | --- |
| `node --check web/dist/app.js` | 通过，退出码0 |
| `.venv/Scripts/python.exe -X utf8 -m py_compile scripts/launch_frontend.py tests/test_frontend_launcher.py` | 通过，退出码0 |
| `.venv/Scripts/python.exe -X utf8 tests/run_regression.py` | 237项通过，27.064秒，退出码0 |
| `scripts/create_frontend_shortcut.ps1` | 退出码0，已生成当前机器的快捷入口 |
| `git diff --check` | 无空白错误；Git提示LF将按已有设置转换CRLF，不是检查失败 |

首次236项回归25.059秒通过；补齐原生单实例检查后237项24.468秒通过；最后调整图片读取序号后完整237项27.064秒通过。没有失败的自动检查。PowerShell将unittest的stderr显示成`NativeCommandError`格式，但进程退出码0且末尾为`OK`；未据此把输出格式误报为测试失败。

回归使用项目测试入口分配的临时目录，原HTTP测试使用替身；它只证明实际覆盖的源代码/进程行为，不证明新页面操作或模型准确率。本轮没有执行真实训练、真实模型前向、数据准备或29/24张封存测试。

## 未验证与限制

- **MANUAL/PENDING**：新启动器窗口、快捷方式双击、新页面的真实浏览器操作、跨窗口拖放、模型切换/重连、GPU推理及下载。本轮未打开、截图或预览新界面，历史证据不能冒充本轮通过。
- 继续依赖本项目已有Python环境和本地模型；快捷入口只适用于当前Windows机器。目录移动后需重建快捷方式。8765端口被占用时先处理此前服务，启动器不接管其他进程。
- 只接受静态单人头像，不自动检测/裁剪头部；分数未经概率校准，模型仍有原有小样本限制。
- 没有Qt工作，也未使用offscreen或其他隐藏渲染作为GUI证据。新增服务后台进程是实际应用运行方式，不是GUI渲染验证。

本轮已授权实施完成，没有新增下一阶段。前端交付提交`bbe102e91f8212919cb5b467f85a52fadd0c9597`已普通推送至origin/main；实际核对本地HEAD与远端main一致、工作区干净。随后仅补充本日志与交接手册中的已发生检查点，独立保存为文档收尾提交，最终推送结果见交付消息。待人工验证项目不会自动触发新训练、打包或额外功能。
