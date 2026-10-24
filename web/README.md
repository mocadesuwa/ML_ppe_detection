# 本机识别前端

`dist/`直接保存页面源码，无打包工具、Node依赖或外部CDN。由`src.serve_demo`同时提供静态资源和本机推理接口。

- `dist/index.html`：上传、模型选择、分数和规则。
- `dist/style.css`：桌面双栏与窄屏布局。
- `dist/app.js`：拖放/选择预览、异步推理、错误处理与JSON下载。

使用项目根目录的`start_demo.cmd`启动，详见[使用说明](../docs/local_demo.md)。不在该目录保存真实头像或模型。
