# UAV World Model & VLA & VLM Learning

<p align="center">
  <a href="README.md">中文</a> | <a href="README.en.md">English</a>
</p>

<p align="center">
  <img src="figures/hero.png" alt="World Model / VLA / VLM Learning Library — for UAVs and embodied AI" width="840">
</p>

<p align="center">
  <a href="https://github.com/Qxy661/UAV-WM-VLA-Learning/stargazers"><img src="https://img.shields.io/github/stars/Qxy661/UAV-WM-VLA-Learning?style=social&label=Star" alt="Star"></a>
  <a href="https://github.com/Qxy661/UAV-WM-VLA-Learning/network/members"><img src="https://img.shields.io/github/forks/Qxy661/UAV-WM-VLA-Learning?style=social&label=Fork" alt="Fork"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Qxy661/UAV-WM-VLA-Learning?color=yellow" alt="License"></a>
  <a href="https://github.com/Qxy661/UAV-WM-VLA-Learning/commits/main"><img src="https://img.shields.io/github/last-commit/Qxy661/UAV-WM-VLA-Learning" alt="Last commit"></a>
</p>

<p align="center">
  <a href="https://qxy661.github.io/UAV-WM-VLA-Learning/"><img src="https://img.shields.io/badge/Docs-Online%20site-0A66C2.svg" alt="Online docs"></a>
  <a href="docs/en/index.md"><img src="https://img.shields.io/badge/Docs-English%20overview-0A66C2.svg" alt="English overview"></a>
  <a href="references/paper-list.md"><img src="https://img.shields.io/badge/Papers-195-b31b1b.svg" alt="195 papers"></a>
  <a href="code/README.md"><img src="https://img.shields.io/badge/Demos-24%20%C2%B7%20CPU--only-181717.svg" alt="24 CPU-only demos"></a>
  <a href="https://arxiv.org/abs/2605.00080"><img src="https://img.shields.io/badge/arXiv-2605.00080-b31b1b.svg" alt="arXiv"></a>
</p>

<p align="center">
  <a href="#what-this-is">What this is</a> •
  <a href="#what-is-different">What is different</a> •
  <a href="#quick-start">Quick start</a> •
  <a href="#contents">Contents</a> •
  <a href="#cite">Cite</a> •
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

---

> **The documentation is written in Chinese.** This file and
> [`docs/en/index.md`](docs/en/index.md) are the English front door: they map the repository and
> link to the Chinese originals. The 59 documents themselves are not translated.

<a id="what-this-is"></a>

## What this is

**59 documents · 195 papers · 24 CPU-only demos · ~287k Chinese characters.**

World models, VLAs and VLMs are the three foundations of embodied AI right now. This repository
turns them into a path you can walk from the beginning: the spine is general method and
literature, and **aerial robotics is a running thread through all nine sections** rather than a
side chapter.

Choosing drones as the specialisation narrows nothing — flight is the harshest embodiment there
is (4-DoF control, a crash you cannot undo, an onboard compute budget measured in watts). Where a
general method breaks in the air is exactly where its limits become visible.

Main references:

- [arXiv:2605.00080](https://arxiv.org/abs/2605.00080) — *World Model for Robot Learning: A
  Comprehensive Survey* (43 pages, 2026).
- [NTUMARS/Awesome-World-Model-for-Robotics-Policy](https://github.com/NTUMARS/Awesome-World-Model-for-Robotics-Policy)
  — a collection of world-model papers for robot policy learning; our annotations are in
  [`references/awesome-annotations.md`](references/awesome-annotations.md).

The three core concepts:

```
VLM (vision-language model)   VLA (vision-language-action)   World Model
"describe what it sees"       "describe, then act"           "predict what comes next"
┌─────────────┐               ┌─────────────┐                ┌─────────────┐
│  image+text │               │ img+text+act│                │ state + act │
│      ↓      │               │      ↓      │                │      ↓      │
│  understand │               │ understand  │                │ predict next│
└─────────────┘               └─────────────┘                └─────────────┘
 "obstacle ahead"            "turn left to avoid"           "what if I turn left"
```

---

<a id="what-is-different"></a>

## What is different

> Paper lists are not scarce. This repository tries to add four things they usually lack: **you
> can run it, check it, track it, and test yourself on it.**

| Asset | What it is |
|-------|------------|
| **24 CPU-only demos** | No training, no VRAM, each produces a figure in about two minutes. Every demo verifies exactly one claim in the text — and where a claim cannot be measured, the text says so instead of implying otherwise. See [`code/`](code/README.md). |
| **Reproducible citation discipline** | [`tools/check_citations.py`](tools/check_citations.py) verifies every arXiv ID in the repository against the official API by title. Results land in [`references/citation-audit.md`](references/citation-audit.md), refreshed by a weekly audit run (see [`CONTRIBUTING.md`](CONTRIBUTING.md)). |
| **Frontier logs for all three topics** | [`tools/watch.py`](tools/watch.py) re-queries world models / VLA / VLM monthly, producing [`references/*-watch-2026-10.md`](references/vla-watch-2026-10.md). |
| **Self-assessment sets** | [`docs/09-专题自测与考察/`](docs/09-专题自测与考察/01-VLA专题自测.md) — questions only; the body text gives pointers, never answers. |
| **A hardware limit stated up front** | Every number comes from a real run. The ceiling is a single 8 GB GPU, so this repository does not train large models — it covers what actually runs. |

A claim in the text looks like this — every figure below comes from a rerunnable script under
`code/`:

<table>
  <tr>
    <td align="center"><img src="figures/g_metrics.png" width="290"><br><sub>Seven navigation metrics, seven different rankings</sub></td>
    <td align="center"><img src="figures/m_exposure_bias.png" width="290"><br><sub>Feed the model its own output: step 60 is 1084× the error of step 1</sub></td>
    <td align="center"><img src="figures/o_budget.png" width="290"><br><sub>VRAM budget: three exact calculations, one measurement</sub></td>
  </tr>
</table>

---

<a id="quick-start"></a>

## Quick start

```bash
git clone https://github.com/Qxy661/UAV-WM-VLA-Learning.git
cd UAV-WM-VLA-Learning
py -3.9 code/a_worldmodel_rssm.py     # CPU only, figure lands in figures/ in seconds
```

On Windows use `py -3.9`; on macOS / Linux use `python3`. All 24 demos need only `numpy` and
`matplotlib` — no GPU, no downloads, no model weights.

If you would rather only read: start with the
[study route](docs/00-导读与学习路线.md) (Chinese) or the
[English documentation overview](docs/en/index.md).

**Read it online:** <https://qxy661.github.io/UAV-WM-VLA-Learning/> — full-text search with
Chinese word segmentation, light/dark themes, rendered diagrams. GitHub Pages can be slow to
reach from mainland China; if it does not load, clone the repository and serve it locally:
`pip install "mkdocs-material==9.7.*" "mkdocs<2" jieba`, then
`py -3.9 tools/build_docs.py && mkdocs serve`.

---

<a id="contents"></a>

## Contents

Nine sections, 59 documents. The Chinese title of each section is the directory name.

| # | Section | Docs | Start |
|---|---------|-----:|-------|
| 00 | Study route and prerequisites | 1 | [导读与学习路线](docs/00-导读与学习路线.md) |
| 01 | Foundations — what a world model / VLM / VLA is | 5 | [什么是世界模型](docs/01-基础概念/01-什么是世界模型.md) |
| 02 | World models — history, generative, model-based RL, 3D, long-horizon | 9 | [世界模型发展史](docs/02-世界模型专题/01-世界模型发展史.md) |
| 03 | VLA — architecture, aerial models, action heads, data, RL post-training | 10 | [VLA 架构演进](docs/03-VLA专题/01-VLA架构演进.md) |
| 04 | VLMs — remote sensing, aerial agents, edge deployment, evaluation | 7 | [遥感 VLM](docs/04-VLM专题/01-遥感VLM.md) |
| 05 | Close reading of one survey, section by section | 5 | [综述概览与结构](docs/05-综述论文精读/01-综述概览与结构.md) |
| 06 | Paper cards — 45 papers, one card each | 4 | [世界模型论文卡片](docs/06-论文导读合集/world-model-papers.md) |
| 07 | Hands-on guides — environment, 10 reproduction guides | 11 | [环境搭建](docs/07-实践指南/01-环境搭建.md) |
| 08 | Research frontier — roadmap, open problems, critical reading | 4 | [研究路线图](docs/08-研究前沿与开放问题/01-研究路线图.md) |
| 09 | Self-assessment — VLA / world model / VLM | 3 | [VLA 专题自测](docs/09-专题自测与考察/01-VLA专题自测.md) |

Also in the repository:

- [`references/paper-list.md`](references/paper-list.md) — all 195 papers with a ★★★ / ★★☆ / ★☆☆
  recommendation scale.
- [`mindmaps/`](mindmaps/field-overview.md) — 4 mind maps (field overview, world-model taxonomy,
  VLA timeline, reading order).
- [`code/`](code/README.md) — the 24 demos.
- [`tools/`](tools/check_citations.py) — citation checking, frontier watching, link and nav linting.

---

<a id="cite"></a>

## Cite

[`CITATION.cff`](CITATION.cff) is in the repository root, so GitHub renders a **Cite this
repository** button with BibTeX and APA exports:

```bibtex
@software{qxy661_uav_wm_vla_2026,
  author  = {Qxy661},
  title   = {UAV World Model \& VLA \& VLM Learning},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/Qxy661/UAV-WM-VLA-Learning},
  license = {MIT}
}
```

## Contributing

The contribution guide is in Chinese: [`CONTRIBUTING.md`](CONTRIBUTING.md). In short — open an
issue with the file and line number, or send a PR that passes `tools/check_links.py`,
`tools/check_nav.py` and `tools/check_citations.py`. Issues are welcome in Chinese or English.

## License

[MIT](LICENSE).
