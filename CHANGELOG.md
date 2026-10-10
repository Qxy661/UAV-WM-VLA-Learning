# 更新日志

本文件记录对外可见的变化。格式参照 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号参照[语义化版本](https://semver.org/lang/zh-CN/)。

## [1.0.0] - 2026-10-10

首个正式版本：59 篇文档、195 篇论文清单、24 个可跑 demo，文档站与英文门面就位。

### 新增

- **文档站**（mkdocs-material）：中文全文检索、mermaid 渲染、明暗双主题、每页「编辑此页」。
- **英文门面**：`README.en.md` 与英文文档总览页。
- **自检工具**：`tools/check_links.py`（全仓相对链接）、`tools/check_nav.py`（nav 与文档一致性）、
  `tools/build_docs.py`（站点暂存树）；前两个已接进 CI。
- **文档站上线**：<https://qxy661.github.io/UAV-WM-VLA-Learning/>（由 `mkdocs gh-deploy`
  发布到 `gh-pages` 分支）。
- **`CITATION.cff`**：GitHub 侧可直接导出 BibTeX / APA 引用格式。
- **三卷自测**（Part 9）：VLA / 世界模型 / VLM 各一篇，正文只给指针不给答案。
- **三卷前沿增量台账**：`references/{vla,wm,vlm}-watch-2026-10.md`。
- Issue 表单（内容错误 / 建议新增论文）与 PR 自查清单。

### 变更

- 定位由「面向无人机的综述学习项目」放宽为**通用的「世界模型 / VLA / VLM」学习库**，
  无人机作为贯穿 9 卷的专章。
- README 首屏重做：hero 图、可点徽章、一句话量化承诺、60 秒快速开始、路线图。
- 论文清单改为条目式排版（arXiv / GitHub / 主页三类徽章）。
- `tools/check_citations.py` 增加 `--fail-on-bad`，可作 CI 门禁（请求失败的号只警告，不判失败）。

### 修复

- 统一「七阶段」口径：README 原写「九阶段」，与导读和文档目录表矛盾。
- README 里指向目录的链接改为指向具体文件，避免在文档站上 404。

## [0.1.0] - 2026-05-10

项目启动。基础概念与世界模型 / VLA / VLM 三卷的初版文档，README、贡献指南与论文清单首版。

[1.0.0]: https://github.com/Qxy661/UAV-WM-VLA-Learning/releases/tag/v1.0.0
