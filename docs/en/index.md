# Documentation Overview (English)

**The body of this repository is written in Chinese.** This page is a map, not a translation:
every document gets an English title and a one-line description, and each entry links to the
Chinese original. If a title here interests you, open the link and read it with a translator —
the figures, tables and code blocks carry most of the argument anyway.

- **59 documents** across nine sections, plus 4 mind maps and 6 reference files.
- **195 papers**, every arXiv ID checked against the arXiv API (see
  [citation audit](../../references/citation-audit.md)).
- **24 CPU-only demos** under [`code/`](../../code/README.md) — no GPU, no training, each runnable
  in about two minutes.
- **Drones are a running thread, not the whole scope.** The spine is general world-model / VLA /
  VLM method and literature; aerial robotics is the specialisation running through all nine
  sections.

[中文首页](../../README.md) | [English README](../../README.en.md)

---

## Start here

- **[导读与学习路线](../../docs/00-导读与学习路线.md)** — Reading paths, prerequisites, and the
  seven-stage route from background to research frontier.

## 1. Foundations

- **[What Is a World Model](../../docs/01-基础概念/01-什么是世界模型.md)** — Definition, history,
  and the four schools that all call themselves "world model".
- **[What Is a VLM](../../docs/01-基础概念/02-什么是VLM.md)** — Vision-language model
  architecture and training in one pass.
- **[What Is a VLA](../../docs/01-基础概念/03-什么是VLA.md)** — How VLA grew out of VLM by adding
  an action head and a control loop.
- **[VLM vs. VLA vs. World Model](../../docs/01-基础概念/04-三者关系与区别.md)** — Where the three
  overlap, where they genuinely differ, and why people conflate them.
- **[Drones vs. Ground Robots](../../docs/01-基础概念/05-无人机vs地面机器人.md)** — What an aerial
  embodiment changes: 6-DoF, tight mass and power budgets, no safe stop, scarce data.

## 2. World Models

- **[A History of World Models](../../docs/02-世界模型专题/01-世界模型发展史.md)** — From
  Ha & Schmidhuber through to modern video world models.
- **[Generative World Models](../../docs/02-世界模型专题/02-生成式世界模型.md)** — Video
  prediction and scene generation for aerial views (ANWM, AirScape, FlightDiffusion).
- **[Model-Based RL World Models](../../docs/02-世界模型专题/03-模型强化学习世界模型.md)** — The
  Dreamer line, and what it takes to fly with it.
- **[3D Scene World Models](../../docs/02-世界模型专题/04-3D场景世界模型.md)** — NeRF and 3D
  Gaussian Splatting read as world models rather than as rendering tricks.
- **[Aerial World Models: Overview](../../docs/02-世界模型专题/05-无人机世界模型综述.md)** — The
  papers that are specifically about UAV world models, perception to decision.
- **[Key Datasets and Benchmarks](../../docs/02-世界模型专题/06-关键数据集与基准.md)** — MotionScape,
  AeroVerse, and what each one can and cannot measure.
- **[Evaluating and Diagnosing World Models](../../docs/02-世界模型专题/07-世界模型评测与诊断.md)** —
  One-step accuracy is not rollout usability; the diagnostics that separate them.
- **[JEPA and Latent World Models](../../docs/02-世界模型专题/08-联合嵌入预测与潜空间世界模型.md)** —
  Predicting in representation space instead of reconstructing pixels, and what that buys.
- **[Long-Horizon and Interactive Generation](../../docs/02-世界模型专题/09-长时程世界模型与交互式生成.md)** —
  Drift, forgetting, and lost object permanence once the horizon passes a hundred steps.

## 3. VLA (Vision-Language-Action)

- **[VLA Architecture Evolution](../../docs/03-VLA专题/01-VLA架构演进.md)** — From RT-2 onward, cut
  along ablation axes rather than a timeline.
- **[Aerial VLA Models](../../docs/03-VLA专题/02-无人机VLA模型.md)** — Purpose-built architectures
  and training recipes (VLA-AN, CognitiveDrone, UAV-Track VLA, AutoFly).
- **[Language-Conditioned Flight Control](../../docs/03-VLA专题/03-语言条件飞行控制.md)** —
  UAV-Flow, VLN-Pilot, and two 2026 aerial vision-language-navigation roadmaps.
- **[Foundation-Model-Assisted Planning](../../docs/03-VLA专题/04-基础模型辅助规划.md)** — CoDrone,
  FM-Planner, NavFoM, FlyMirage: where the LLM sits in the loop.
- **[Onboard Deployment](../../docs/03-VLA专题/05-机载部署与优化.md)** — Inference speedups, edge
  budgets, and action age as the real lower bound on control rate.
- **[Action Heads and Action Chunking](../../docs/03-VLA专题/06-动作头与动作分块.md)** — Discrete
  tokens, regression, diffusion, flow matching — and how to pick chunk length.
- **[Data, Pretraining and Cross-Embodiment](../../docs/03-VLA专题/07-数据、预训练与跨具身.md)** —
  OXE/DROID, co-training, latent actions; flight data scarcity as the first constraint.
- **[RL Post-Training and Self-Improvement](../../docs/03-VLA专题/08-强化学习后训练与自我改进.md)** —
  GRPO/PPO on top of a VLA, where rewards come from, and how reward hacking shows up.
- **[Benchmarks and Reporting](../../docs/03-VLA专题/09-评测基准与报告口径.md)** — LIBERO to
  RoboArena; the three choices hiding behind a single success-rate number.
- **[World-Model-Augmented VLA](../../docs/03-VLA专题/10-世界模型增强VLA.md)** — A world model as
  data source, policy component, simulator, or evaluation proxy — four different claims.

## 4. VLMs

- **[Remote-Sensing VLMs](../../docs/04-VLM专题/01-遥感VLM.md)** — GeoChat, RSGPT, SkySenseGPT and
  why overhead imagery breaks ordinary VLMs.
- **[Aerial Scene Understanding Benchmarks](../../docs/04-VLM专题/02-无人机场景理解.md)** — UAVBench,
  BEDI, and what "understanding" is actually scored as.
- **[LLM-Powered UAV Agents](../../docs/04-VLM专题/03-LLM驱动的无人机Agent.md)** — CityNavAgent,
  ACDC: language models as planners and tool users for flight.
- **[Edge VLM Deployment](../../docs/04-VLM专题/04-边缘VLM部署.md)** — Lightweighting,
  distillation, BLIP-2, and what survives an 8 GB budget.
- **[General VLM Architecture and Vision Encoders](../../docs/04-VLM专题/05-通用VLM架构与视觉编码器.md)** —
  Four components, CLIP to native resolution, connectors, and the token budget.
- **[VLM Instruction Tuning and Alignment](../../docs/04-VLM专题/06-VLM指令微调与对齐.md)** — Building
  instruction data, the limits of LoRA, frozen versus joint training.
- **[VLM Evaluation and Hallucination](../../docs/04-VLM专题/07-通用VLM评测与幻觉.md)** — MMMU,
  MMBench, POPE, plus negative sampling and yes-bias.

## 5. Survey Close-Read

A five-part close reading of a single survey (arXiv:2605.00080), section by section.

- **[Survey Overview and Structure](../../docs/05-综述论文精读/01-综述概览与结构.md)**
- **[World Model as Policy](../../docs/05-综述论文精读/02-世界模型作为策略.md)**
- **[World Model as Simulator](../../docs/05-综述论文精读/03-世界模型作为模拟器.md)**
- **[Video Generation World Models](../../docs/05-综述论文精读/04-视频生成世界模型.md)**
- **[Benchmarks and Evaluation](../../docs/05-综述论文精读/05-基准与评估.md)**

## 6. Paper Cards

Short annotated cards — one per paper, with the claim, the setup, and the caveat.

- **[World-Model Paper Cards](../../docs/06-论文导读合集/world-model-papers.md)** — 13 papers.
- **[VLA Paper Cards](../../docs/06-论文导读合集/vla-papers.md)** — 11 papers.
- **[VLM Paper Cards](../../docs/06-论文导读合集/vlm-papers.md)** — 12 papers.
- **[Benchmark and Dataset Papers](../../docs/06-论文导读合集/benchmark-papers.md)** — 9 papers.

## 7. Hands-On Guides

- **[Environment Setup](../../docs/07-实践指南/01-环境搭建.md)** — CUDA, PyTorch, ROS2, and the
  version pairings that actually work.
- **[Reproducing UAV-Flow](../../docs/07-实践指南/02-复现指南-UAV-Flow.md)** — Fine-grained
  language-conditioned drone control in the real world.
- **[Reproducing CognitiveDrone](../../docs/07-实践指南/03-复现指南-CognitiveDrone.md)** — Data
  collection and benchmark for cognitive UAV tasks.
- **[Reproducing FlightDiffusion](../../docs/07-实践指南/04-复现指南-FlightDiffusion.md)** —
  Diffusion-generated FPV video, plus two corrections to the paper.
- **[Reproducing MotionScape](../../docs/07-实践指南/05-复现指南-MotionScape.md)** — A benchmark for
  future-video generation from a drone's point of view.
- **[Reproducing GeoChat](../../docs/07-实践指南/06-复现指南-GeoChat.md)** — Remote-sensing VLM.
- **[Reproducing DreamerV3-Drone](../../docs/07-实践指南/07-复现指南-DreamerV3-Drone.md)** —
  World-model-based drone racing.
- **[Reproducible Project Candidates](../../docs/07-实践指南/08-可复现项目候选清单.md)** — 17
  projects that one person with one machine can actually run.
- **[Reproducing DeepDrone](../../docs/07-实践指南/09-复现指南-DeepDrone.md)** — Natural-language
  drone control with an LLM.
- **[Reproducing RemoteCLIP](../../docs/07-实践指南/10-复现指南-RemoteCLIP.md)** — A remote-sensing
  vision-language foundation model.
- **[Reproducing Flightmare](../../docs/07-实践指南/11-复现指南-Flightmare.md)** — A high-fidelity
  drone simulator.

## 8. Research Frontier

- **[Research Roadmap](../../docs/08-研究前沿与开放问题/01-研究路线图.md)** — A five-step path from
  learner to researcher, with the FINER criteria applied at each step.
- **[Open Problems](../../docs/08-研究前沿与开放问题/02-研究空白与机会.md)** — Where aerial VLA, VLM
  and world models are genuinely under-served.
- **[Critical Reading](../../docs/08-研究前沿与开放问题/03-论文批判性阅读.md)** — The CRITIC method
  and a reading-note template.
- **[Latest Progress and Lab Tracking](../../docs/08-研究前沿与开放问题/04-最新进展与团队追踪.md)** —
  2025–2026 progress, and how to keep tracking it yourself.

## 9. Self-Assessment

Questions only — the body text gives pointers, never answers.

- **[VLA Self-Assessment](../../docs/09-专题自测与考察/01-VLA专题自测.md)** — Three tiers of
  questions plus 12 aerial-specific ones.
- **[World-Model Self-Assessment](../../docs/09-专题自测与考察/02-世界模型专题自测.md)**
- **[VLM Self-Assessment](../../docs/09-专题自测与考察/03-VLM专题自测.md)**

## Mind Maps

- **[Field Overview](../../mindmaps/field-overview.md)** — How world models, VLA and VLM relate.
- **[World-Model Taxonomy](../../mindmaps/world-model-taxonomy.md)**
- **[VLA Evolution Timeline](../../mindmaps/vla-evolution.md)**
- **[Recommended Reading Order](../../mindmaps/reading-order.md)**

## References

- **[Full Paper List](../../references/paper-list.md)** — 195 entries with a ★★★ / ★★☆ / ★☆☆
  recommendation scale.
- **[VLA Watch, 2026-10](../../references/vla-watch-2026-10.md)** — Incremental frontier log.
- **[World-Model Watch, 2026-10](../../references/wm-watch-2026-10.md)**
- **[VLM Watch, 2026-10](../../references/vlm-watch-2026-10.md)**
- **[Awesome-List Annotations](../../references/awesome-annotations.md)** — Notes on
  NTUMARS/Awesome-World-Model-for-Robotics-Policy and what we add on top.
- **[Citation Audit Report](../../references/citation-audit.md)** — Every arXiv ID checked against
  the arXiv API, with the failures listed.

---

## How the numbers stay trustworthy

Two rules run through the whole repository, and they are enforced by tooling rather than by good
intentions:

1. **Every arXiv ID comes from the arXiv API.** [`tools/check_citations.py`](../../tools/check_citations.py)
   re-verifies all of them and regenerates the audit report; the audit is re-run weekly. A paper
   that cannot be found is reported as not found — never silently kept.
2. **Every number comes from a real run.** The demos under [`code/`](../../code/README.md) are
   CPU-only and take about two minutes each; if a figure appears in the text, the script that
   produced it is in the repository.

Known limit, stated rather than hidden: the reference hardware is a single laptop GPU with **8 GB
of VRAM**, so nothing here trains a world model or a VLA from scratch. Anything that would need
more is marked as such instead of being faked.

## Citation

See [`CITATION.cff`](../../CITATION.cff) in the repository root — GitHub renders it as a
"Cite this repository" button with BibTeX and APA exports.

## Contributing

The full contribution guide is in Chinese: [`CONTRIBUTING.md`](../../CONTRIBUTING.md). In short —
open an issue with the file and line, or send a PR that passes `tools/check_links.py`,
`tools/check_nav.py` and `tools/check_citations.py`.
