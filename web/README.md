# 本机识别前端

`dist/`直接保存页面源码，无打包工具、Node依赖或外部CDN。由`src.serve_demo`同时提供静态资源和本机推理接口。

- `dist/index.html`：以头像上传、模型选择和识别结果为主体；模型资料默认收起，通过三个小按钮打开。
- `dist/style.css`：Neutral Modern浅灰/钴蓝样式、桌面双栏与窄屏布局、可见焦点和减少动态效果支持。
- `dist/app.js`：拖放/选择预览、清空、异步推理、重连、错误处理与JSON下载。
- `dist/training.js`、`dist/training-summary.json`：资料面板展开/收起、四组已保存实验的曲线切换、验证对照与历史测试说明；摘要只含聚合数据和证据哈希，展开曲线后按容器宽度绘制。

使用项目根目录的`护目镜识别.lnk`打开图形启动器；入口由`scripts/create_frontend_shortcut.ps1`在本机生成，调用现有虚拟环境中的pythonw与`scripts/launch_frontend.py`。原`start_demo.cmd`保留为备用入口，详见[使用说明](../docs/local_demo.md)。不在该目录保存真实头像或模型。
