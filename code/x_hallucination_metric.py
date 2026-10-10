"""
Demo X — 幻觉评测：负样本怎么选，决定了谁排第一

对应文档：docs/04-VLM专题/07-通用VLM评测与幻觉.md

要验证的结论：POPE 式幻觉评测问的是「图里有没有 X」这种二分类问题，指标（Accuracy /
F1）看着客观，但**换一套负样本，名次会翻**。三个模型：

  M_prior    图像盲：只按"这个类常不常见"回答，一眼都不看图。它是语言先验的捷径。
  M_mixed    一半回合靠先验捷径、一半靠视觉接地的混合体（现实的中间态）。
  M_vision   有视觉接地，但判别力有限。

负样本每档 500 题，X 都不在图上，区别只在**怎么挑 X**：
  random      在词表上**逐类等概率**抽 —— 抽到的大多是冷门类，语言先验答"没有"就对了
  popular     按类频的 1.5 次方抽 —— 多半是常见类，恰巧这张图没有
  adversarial 从与图上物体最相似的类里、再按类频 1.5 次方抽 —— 先验说"应该有"、图上没有

正题（图上确实有的物体）按类频的 1.5 次方抽：标注数据里常见类本来就占主导。

三条实测结论：
  ① random 档上，图像盲的 M_prior 排第一——这一档**分辨不出模型看没看图**：
     正题靠先验就能答对，冷门负题答"没有"也对。
  ② 换成 popular / adversarial 负样本，名次翻过来，M_vision 第一。翻转的原因不是
     模型变了，是负样本的抽样分布变了。
  ③ yes-ratio 比 Accuracy 更早暴露"它在猜 yes"。

装置（概率模型 + 蒙特卡洛，不训练任何模型）：
  词表 V=1000，类频 Zipf(alpha=1)。M_prior 的阈值：类序号 < 100 就答 yes。
  M_vision 的判正率固定 0.85，判负率按负样本难度取 random 0.90 / popular 0.80 /
  adversarial 0.62 —— **这三个数是本模型的假设输入，不是学出来的**，见"外推边界"。
  另有第 ③ 节扫描正题的类频集中度 beta，看翻转在什么条件下出现。

自检点：
  (1) M_prior 在 random 档的 Accuracy 应高于 M_vision（图像盲也能排第一）。
  (2) M_prior 在 popular / adversarial 档应掉到 0.55 以下（近乎全答 yes）。
  (3) 名次应在 random 与 popular 之间翻转。
  (4) M_prior 的 yes-ratio 应随负样本变难而单调升到接近 1.0。

运行：py -3.9 code/x_hallucination_metric.py
纯 CPU、纯 numpy，约 2 秒；不训练任何模型。
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import plotting

SEED = 21
VOCAB = 1000                     # 词表大小
ALPHA = 1.0                      # Zipf 指数：p_i ∝ 1/i^ALPHA
BETA = 1.5                       # 正题与"常见类负样本"的类频集中度：∝ p^BETA
N_PER_CLASS = 500                # 每档正/负题的题数
RANK_THRESHOLD = 100             # M_prior 的阈值：类序号 < 100 就答 yes
P_VIS_RECALL = 0.85              # M_vision 判"在"的概率（假设输入）
TN_BY_SPLIT = {"random": 0.90, "popular": 0.80, "adversarial": 0.62}
SPLITS = ["random", "popular", "adversarial"]
K_SIM = 50                       # adversarial：取最相似的 K 个类做候选池
BETAS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]


def zipf(alpha=ALPHA):
    """类频 p_i ∝ 1/i^alpha，已归一化。"""
    w = 1.0 / np.arange(1, VOCAB + 1) ** alpha
    return w / w.sum()


def weighted_cum(p, beta):
    """按 p^beta 加权的类别累计分布。"""
    w = p ** beta
    return np.cumsum(w / w.sum())


def build_questions(rng, beta=BETA):
    """三个档位的题库：{split: (queried_class, truth)}，正负各一半。"""
    p = zipf()
    cum_b = weighted_cum(p, beta)
    emb = rng.standard_normal((VOCAB, 8))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    present = np.searchsorted(cum_b, rng.random(N_PER_CLASS))
    # adversarial 候选池：与每个正题类最相似的 K 个类，池内再按 p^beta 抽
    sim = emb @ emb[present].T
    topk = np.argpartition(-sim, K_SIM, axis=0)[:K_SIM]        # (K, N)
    w = p[topk] ** beta
    w /= w.sum(axis=0, keepdims=True)
    u = rng.random(N_PER_CLASS)[None, :]
    adv = topk[(np.cumsum(w, axis=0) > u).argmax(axis=0), np.arange(N_PER_CLASS)]
    absent = {
        "random": rng.integers(0, VOCAB, N_PER_CLASS),          # 逐类等概率
        "popular": np.searchsorted(cum_b, rng.random(N_PER_CLASS)),
        "adversarial": adv,
    }
    truth = np.concatenate([np.ones(N_PER_CLASS, bool),
                            np.zeros(N_PER_CLASS, bool)])
    return {s: (np.concatenate([present, absent[s]]), truth) for s in SPLITS}


def answer_prior(q):
    """图像盲：只看类序号，常见类答 yes。"""
    return q < RANK_THRESHOLD


def answer_vision(q, truth, rng, tn, mix=0.0):
    """有视觉接地：正题按 recall 判对，负题按 tn 判对；mix>0 时按比例退回先验。"""
    hit = np.where(truth, rng.random(len(q)) < P_VIS_RECALL,
                   rng.random(len(q)) < tn)
    vis = np.where(hit, truth, ~truth)
    if mix > 0.0:
        swap = rng.random(len(q)) < mix
        vis = np.where(swap, answer_prior(q), vis)
    return vis


def metrics(pred, truth):
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum()); tn = int((~pred & ~truth).sum())
    acc = (tp + tn) / len(truth)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return acc, f1, prec, rec, float(pred.mean())


def eval_models(rng, bank, models):
    res = {}
    for split in SPLITS:
        q, truth = bank[split]
        for name in models:
            if name == "M_prior":
                pred = answer_prior(q)
            elif name == "M_mixed":
                pred = answer_vision(q, truth, rng, TN_BY_SPLIT[split], mix=0.5)
            else:
                pred = answer_vision(q, truth, rng, TN_BY_SPLIT[split], mix=0.0)
            res[(split, name)] = metrics(pred, truth)
    return res


def beta_scan(main_res):
    """扫正题集中度 beta：M_prior 在 random 档还领先吗？

    beta = BETA 那一行直接取主实验（①）的结果，避免同一实验因重抽而差出噪声。
    """
    models = ["M_prior", "M_vision"]
    rows = []
    for b in BETAS:
        if b == BETA:
            r = main_res
        else:
            bank = build_questions(np.random.default_rng(SEED + int(b * 1000)), b)
            r = eval_models(np.random.default_rng(SEED + 5), bank, models)
        rows.append((b, r[("random", "M_prior")][0], r[("random", "M_vision")][0],
                     r[("popular", "M_prior")][0], r[("popular", "M_vision")][0]))
    return rows


def main():
    font = plotting.use_chinese_font()
    print("=" * 78)
    print("Demo X - 幻觉评测：负样本怎么选，决定了谁排第一")
    print("=" * 78)
    print(f"字体: {font or '未找到 CJK 字体'}")
    p = zipf()
    print(f"词表 {VOCAB} 类，类频 Zipf(alpha={ALPHA})：最常见类 p={p[0]:.4f}")
    print(f"每档 {N_PER_CLASS} 正 + {N_PER_CLASS} 负。正题与常见负样本按 p^{BETA} 抽。")
    print(f"M_prior 阈值 = 类序号 < {RANK_THRESHOLD} 答 yes；"
          f"M_vision 判正率 {P_VIS_RECALL}，判负率 {TN_BY_SPLIT}")

    models = ["M_prior", "M_mixed", "M_vision"]
    rng = np.random.default_rng(SEED)
    bank = build_questions(rng)
    res = eval_models(rng, bank, models)

    print("\n" + "-" * 78)
    print("① 三档负样本上的 Accuracy / F1 / Precision / Recall / yes-ratio")
    print("-" * 78)
    desc = {"random": "全词表逐类等概率（多是冷门类）",
            "popular": f"按类频 p^{BETA}（常见类）",
            "adversarial": f"最相似 {K_SIM} 类内按 p^{BETA}"}
    for split in SPLITS:
        print(f"[{split}]  负样本 = {desc[split]}")
        print(f"{'模型':>10}{'Accuracy':>11}{'F1':>9}{'Precision':>11}"
              f"{'Recall':>9}{'yes-ratio':>11}")
        for name in models:
            m = res[(split, name)]
            print(f"{name:>10}{m[0]:>11.4f}{m[1]:>9.4f}{m[2]:>11.4f}"
                  f"{m[3]:>9.4f}{m[4]:>11.4f}")
        order = sorted(models, key=lambda n: -res[(split, n)][0])
        print("   名次（按 Accuracy）: " + " > ".join(order))

    print("\n" + "-" * 78)
    print("② random 档分辨不出「看没看图」，名次在 random 与 popular 之间翻转")
    print("-" * 78)
    print(f"{'负样本抽样':>14}{'M_prior':>10}{'M_mixed':>10}{'M_vision':>10}"
          f"{'第一':>10}{'M_prior yes-ratio':>20}")
    for split in SPLITS:
        a = {n: res[(split, n)][0] for n in models}
        print(f"{split:>14}{a['M_prior']:>10.4f}{a['M_mixed']:>10.4f}"
              f"{a['M_vision']:>10.4f}{max(a, key=a.get):>10}"
              f"{res[(split, 'M_prior')][4]:>20.4f}")
    r1, r2 = res[("random", "M_prior")][0], res[("popular", "M_prior")][0]
    print(f"   -> 图像盲的 M_prior 在 random 档拿到 {r1:.4f} 并排第一，换成 popular 负样本")
    print(f"      掉到 {r2:.4f}（差 {r1 - r2:.4f}）。同一套词表、同一个阈值、同一个模型，")
    print(f"      差别只在负样本怎么抽。它的 Recall 在三档上都是 "
          f"{res[('random', 'M_prior')][3]:.4f}——正题靠先验就答完了。")
    print(f"   -> M_mixed 对 M_vision：random 档 "
          f"{res[('random', 'M_mixed')][0] - res[('random', 'M_vision')][0]:+.4f}"
          f"（混合体赢），popular 档 "
          f"{res[('popular', 'M_mixed')][0] - res[('popular', 'M_vision')][0]:+.4f}"
          f"（接地模型赢）。**名次翻转**。")

    print("\n" + "-" * 78)
    print("③ 稳健性：翻转依赖正题的类频集中度 beta 吗？")
    print("-" * 78)
    print(f"{'beta':>6}{'M_prior(R)':>13}{'M_vision(R)':>14}{'random 谁第一':>16}"
          f"{'M_prior(P)':>13}{'M_vision(P)':>14}{'popular 谁第一':>16}")
    rows = beta_scan(res)
    for b, mp, mv, mpp, mvp in rows:
        print(f"{b:>6.2f}{mp:>13.4f}{mv:>14.4f}"
              f"{('M_prior' if mp > mv else 'M_vision'):>16}"
              f"{mpp:>13.4f}{mvp:>14.4f}"
              f"{('M_prior' if mpp > mvp else 'M_vision'):>16}")
    flips = [b for b, mp, mv, _, _ in rows if mp > mv]
    print(f"   -> 正题越集中在常见类上（beta 越大），图像盲模型在 random 档越占便宜：")
    print(f"      翻转出现在 beta >= {min(flips):.2f} 时" if flips else
          "      本扫描区间内未出现翻转")
    print(f"      （beta={BETAS[0]:.2f} 时 random 档 M_prior 只有 {rows[0][1]:.4f}，排序与"
          f" beta={BETAS[-1]:.2f} 相反）。")
    print(f"      beta={BETA:.2f} 那一行就是 ① 的主实验（不重抽）；M_vision 行近乎水平，"
          f"因为它不看类频先验，")
    print(f"      beta 只通过取样噪声影响它。")
    print("      所以「random 档会高估先验模型」这条结论**有条件**：条件就是真实标注数据里")
    print("      正题类的类频集中程度。那个数要从真实数据统计，本 demo 不能替代。")
    print("      不依赖 beta 的那一条是：popular 档下 M_prior 一律垫底。")

    print("\n" + "-" * 78)
    print("④ yes-ratio 是更早暴露问题的那个数")
    print("-" * 78)
    print(f"{'模型':>10}" + "".join(f"{s:>16}" for s in SPLITS))
    for name in models:
        print(f"{name:>10}" + "".join(f"{res[(s, name)][4]:>16.4f}" for s in SPLITS))
    print(f"   -> M_prior 的 yes-ratio 从 random 档 {res[('random', 'M_prior')][4]:.4f} 升到")
    print(f"      popular 档 {res[('popular', 'M_prior')][4]:.4f}，逼近全答 yes。Accuracy 只掉到")
    print(f"      0.5 附近，yes-ratio 却把「答对了没有」和「有没有在猜」分开了。")
    print("   -> 只报 Accuracy 的话，random 档会得出「图像盲模型最强」的结论。")

    print("\n" + "-" * 78)
    print("外推边界（必须与结论一起读）")
    print("-" * 78)
    print("   本实验是**决策规则的概率模型**：类频、相似类结构、M_vision 的三个判负率")
    print("   都是设定的，不是从真实 VLM 上测出来的。它量的是「评测协议对负样本分布的")
    print("   敏感性」，不是某个模型的幻觉水平。")
    print("   结论里**不依赖**那些设定值的部分是：popular 档下图像盲模型一律垫底；")
    print("   同一协议下，任何掺先验的模型都会被「能靠先验答对的负样本」高估。")
    print("   已知不能外推的部分：(a) 真实模型的 yes-偏置随问题类型变，不是常数；")
    print("   (b) 真实共现结构来自数据统计，这里用随机嵌入近似；(c) 真实基准的正题")
    print(f"   分布比 p^{BETA} 更复杂，翻转的具体位置取决于它；(d) {N_PER_CLASS} 题的二项")
    print("   标准误约 1.6 个百分点，档间小于 2 个百分点的比较不可靠。")

    # ---- 出图 ----
    fig, axes = plotting.plt.subplots(1, 3, figsize=(15.2, 4.6))
    xs = np.arange(len(SPLITS))
    cols = {"M_prior": "#C44E52", "M_mixed": "#8C8C8C", "M_vision": "#4C72B0"}
    w = 0.26

    ax = axes[0]
    for i, nm in enumerate(models):
        vals = [res[(s, nm)][0] for s in SPLITS]
        ax.bar(xs + (i - 1) * w, vals, width=w, color=cols[nm], label=nm)
        for x_, v in zip(xs + (i - 1) * w, vals):
            ax.text(x_, v + 0.01, f"{v:.3f}", ha="center", fontsize=6.5)
    ax.set_xticks(xs); ax.set_xticklabels(SPLITS, fontsize=8)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8, loc="lower left")
    ax.set_title("(1) random 档图像盲模型第一")

    ax = axes[1]
    bs = [r[0] for r in rows]
    ax.plot(bs, [r[1] for r in rows], "o-", color=cols["M_prior"], label="M_prior (random)")
    ax.plot(bs, [r[2] for r in rows], "s-", color=cols["M_vision"], label="M_vision (random)")
    ax.plot(bs, [r[3] for r in rows], "o--", color=cols["M_prior"], alpha=0.5,
            label="M_prior (popular)")
    ax.set_xlabel("正题的类频集中度 beta"); ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc="center left")
    ax.set_title("(2) 翻转的条件：正题够不够集中在常见类")

    ax = axes[2]
    for nm in models:
        ax.plot(xs, [res[(s, nm)][4] for s in SPLITS], "o-", color=cols[nm], label=nm)
    ax.axhline(0.5, color="#8C8C8C", ls="--", lw=1)
    ax.text(0.02, 0.52, "全答 yes 的 50% 线", fontsize=7, color="#8C8C8C")
    ax.set_xticks(xs); ax.set_xticklabels(SPLITS, fontsize=8)
    ax.set_ylabel("yes-ratio"); ax.set_ylim(0, 1.08)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("(3) yes-ratio：先验模型逼近 1.0")

    fig.suptitle("同一个模型，换一套负样本就换一个名次", y=1.0)
    fig.tight_layout()
    plotting.save(fig, "x_hallucination_metric.png")


if __name__ == "__main__":
    main()
