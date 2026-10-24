# 更新日志

本文件记录对外可见的变化。格式参照 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号参照[语义化版本](https://semver.org/lang/zh-CN/)。

## [未发布]

知识层重构：把散在各篇里的"结论"收成一张可校验的知识点图，并补一卷跨卷判据。

### 新增

- **知识点注册表**（`references/knowledge-points.md`）：134 条知识点的唯一真源，带 ID、层级、前置、判据、
  承载文档与可验证代码。「前置」列构成有向图，`tools/kp.py check` 拒绝成环、无判据、未登记的条目。
- **头部五字段**：`定位 / 本篇 KP / 读完后应能 / 判据 / 预计阅读`，59 篇正文全部迁移（`en/index.md` 为导航页，
  以 `kp-exempt` 注释豁免）。
- **Part 10 · 横切判据**（`docs/10-横切判据/`）：十篇跨卷的尺子——口径不可横比、算力与延迟预算、
  带宽到频率换算、动作表示选择、状态观测信念、数据泄漏与划分、世界模型诊断、安全层与降级、
  训练工程最小集、导出与尾部延迟。每篇一张判据表、一个反例、一次对本库既有 demo 的指认。
- **`tools/kp.py`**：`derive`（从 H2 抽候选）、`check`（注册表 + 头部校验）、`stats`（覆盖率）。
- **工程层骨架**（`projects/uav-inspect/`）：三机协同巡检的分层频率设计与 M0 悬停验收（尚未验证）。
- **数字对账**（`tools/check_numbers.py` + `code/expected/metrics.json`）：参与核对的六个 demo 用
  `common.metrics.emit` 打一行 `[metrics] k=v`，CI 单独一个 job 重跑并与基线逐项比（相对容差 1e-3）。
- **论文卡片回指索引**：四册共 45 张卡片各配一张「回到正文」表，指向正文里讲它的那一节；
  45 张里有 5 张在正文中没有落位，表里如实标出，不再让读者以为卡片等于体系。

### 变更

- 世界模型卷按依赖重排：评测与诊断升到 06（先立判据），关键数据集与基准降到 07；VLM 卷把通用主干
  01–03 前置、应用 04–07 后置。导航、正文交叉引用与「第 NN 篇」表述同步。
- README 首屏由"多做三件事"改为四件，文档目录新增 Part 10 与注册表入口。
- 三个 demo 换了能站住脚的对照：`e_gsc_safety` 同场景开/关共用种子（凸形 100%→0、凹形 50.0%→77.0%），
  `c_action_heads` 两个头对齐为 256 宽 / 8000 步（参数量 77.3k 对 101.9k），`b_lang_cond_policy` 改报
  hold-out 损失（0.00203→0.09763）并配上闭环误差（1.89→2.15 m）。相关文档里的旧数字与旧结论一并更正。

### 修复

- 修正 VLM 卷导读里"先读 01/02"与重排后编号不符的推荐顺序。
- 修掉六个静默 bug：`r_eval_protocol` 抽样池越界、`x_hallucination_metric` 对抗档负样本混入正题类、
  `t_wm_diagnostics` 打印恒为 0 的净增量、`n_quant_bits` 图例与判据不一致、`u_jepa_probe` 测试集被截断、
  `a_worldmodel_rssm` 的 KL 方向混用与"开环"实际每步喂真观测。
- 修掉五处转写产生的事实冲突：LLaVA-Instruct 的 41K 应为 77K（合计才对得上 15.8 万）、
  DreamerV3 优化器由"论文未明确"改为 LaProp + AGC、free bits 改为整条 KL 的下限 1 nat（原先按维度相乘）、
  `SkyAgentEval` 统一为 `SkyAgent-Eval`、Dream to Fly 的会议年份以论文清单为准；另把
  "Jetson Nano 472 GFLOPS ~ Orin Nano 40 TOPS"标明是两种口径、4090 的 450 W 从 50–200 W 档里拎出来。
- 全量重跑的耗时口径更新为 765.9 秒（原 944.8 秒），`code/README.md` 的单脚本估计改为实测值。

## [1.0.0] - 2026-10-10

首个正式版本：59 篇文档、195 篇论文清单、24 个可跑 demo，文档站与英文门面就位。

### 新增

- **文档站**（mkdocs-material）：中文全文检索、mermaid 渲染、明暗双主题、每页「编辑此页」。
- **英文门面**：`README.en.md` 与英文文档总览页。
- **自检工具**：`tools/check_links.py`（全仓相对链接）、`tools/check_nav.py`（nav 与文档一致性）、
  `tools/build_docs.py`（站点暂存树）；前两个已接进 CI。
- **文档站上线**：<https://qxy661.github.io/UAV-WM-VLA-Learning/>（由 `mkdocs gh-deploy`
  发布到 `gh-pages` 分支）。
- **持续集成**：`.github/workflows/ci.yml`（相对链接 / nav / 英文总览三项检查，main 上构建并发布站点）。
- **`CITATION.cff`**：GitHub 侧可直接导出 BibTeX / APA 引用格式。
- **三卷自测**（Part 9）：VLA / 世界模型 / VLM 各一篇，正文只给指针不给答案。
- **三卷前沿增量台账**：`references/{vla,wm,vlm}-watch-2026-10.md`。
- Issue 表单（内容错误 / 建议新增论文）与 PR 自查清单。

### 变更

- 定位由「面向无人机的综述学习项目」放宽为**通用的「世界模型 / VLA / VLM」学习库**，
  无人机作为贯穿 9 卷的专章。
- README 首屏重做：hero 图、可点徽章、一句话量化承诺、60 秒快速开始、路线图。
- 论文清单改为条目式排版（arXiv / GitHub / 主页三类徽章）。

### 修复

- 统一「七阶段」口径：README 原写「九阶段」，与导读和文档目录表矛盾。
- README 里指向目录的链接改为指向具体文件，避免在文档站上 404。

## [0.1.0] - 2026-05-10

项目启动。基础概念与世界模型 / VLA / VLM 三卷的初版文档，README、贡献指南与论文清单首版。

[1.0.0]: https://github.com/Qxy661/UAV-WM-VLA-Learning/releases/tag/v1.0.0
