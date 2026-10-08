# 02：坐标转换

- 提交时间：2026-10-08 11:01:49 +08:00；关联提交：`5aad545`。
- 记录方式：2026-10-08 根据 Git、版本记录与测试输出事后补录。
- 文件：`src/prepare_dataset.py`、`tests/test_data_pipeline.py`、`README.md`、`CHANGELOG.md`、`docs/framework_review.md`。

## 原因与修改

原有三项坐标测试通过，但无限图片宽度可返回零宽 YOLO 标注。添加正整数图片尺寸校验，拒绝小数、无限值、布尔值及其他无效尺寸；补充 VOC 单像素边界框和未知坐标约定测试，两种原有坐标约定保留。

## 验证与回顾

- 新增测试在修复前复现失败；修复后运行 `.\.venv\Scripts\python.exe -m unittest tests.test_data_pipeline.BoxTests -v`，6 项全部通过，包含原有三项。
- `git diff --check` 与暂存差异检查通过，提交已上传 GitHub。
- 本轮只验证坐标组，未宣称完整流程通过；`.venv` 的 Pillow 缺失等问题仍存在。
- 没有安装依赖、处理真实数据或启动训练。

## 后续

检查 `split_groups` 与 `group_samples`，同时复查坐标测试。
