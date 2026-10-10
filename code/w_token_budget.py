"""
Demo W — 视觉 token 预算：压下去，先崩的是定位，不是描述

对应文档：docs/04-VLM专题/05-通用VLM架构与视觉编码器.md

要验证的结论：视觉 token 数是**分辨率**的代理量。预算压到 B 个 token，格子边长
s = 1/G = 1/sqrt(B) 变大，同一个量化误差在三类处理方式下变成三种东西：

  整数统计（计数、主导区域）  格占用数求和。切成 64 格还是 4 格，结果一样 ——
                              **精确不变**，与 B 无关。
  场景聚合（质心）            逐目标误差是零均值的，N 个目标一平均，误差降 ~sqrt(N)，
                              即 ∝ s/sqrt(N) —— 与逐目标同斜率，但低 sqrt(N) 倍。
  依赖坐标的判读（谁在左边、目标在哪）  误差 ∝ s = 1/sqrt(B)，**指数 -1/2**。

所以能站得住的说法是**两层**，不是三层：**任务是"对整数求和"还是"读坐标"，决定它
怕不怕压预算**；只要需要坐标，"说谁在左边"和"指出它在哪"走的是同一条曲线（常数
不同：0.5/sqrt(B) 对 0.3826/sqrt(B)），B 除以 144，两者都涨 12 倍。
这正是 token 裁剪、token 合并一类方法必须先声明自己牺牲了哪一类的原因。

装置（纯几何，不训练任何模型）：
  一"张"图是单位正方形，N_OBJ 个目标位置 ~ U([0,1]^2) 独立同分布。
  视觉编码器被建模成一个 G x G 的均匀网格（G = round(sqrt(B))），每格出一个 token，
  token 携带的信息 = 落进该格目标的**格心坐标**。下游只从 token 读信息。

自检点：
  (1) 计数与主导象限的正确率应恒为 1.0000（整数求和，与分区无关）。
  (2) 质心误差应比逐目标定位误差低约 sqrt(N_OBJ) = 8 倍这一档。
  (3) 定位平均误差贴闭式 (s/6)(sqrt(2)+ln(1+sqrt(2))) = 0.382598*s，RMS 贴 s/sqrt(6)。
  (4) 左右判断的失手率贴 0.5/G：只依赖一个坐标列，"不同列"的概率是 1/G 而不是 1/B。
  (5) 容差命中率贴闭式 pi*tau^2/s^2，即**与 B 成正比**（B 翻 4 倍，命中率约翻 4 倍）。
  (6) 把 token 集中给目标所在象限：该象限误差降 ~1.5 倍，其余三区各差 ~1.33 倍，
      按面积加权**整体平均反而更差 ~1.17 倍**。

运行：py -3.9 code/w_token_budget.py
纯 CPU、纯 numpy，约 3 秒；不训练任何模型。
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import plotting

SEED = 13
BUDGETS = [576, 256, 64, 36, 16, 4]     # 视觉 token 数，自变量
N_SCENE = 400                            # 场景数（每个场景重采目标位置）
N_OBJ = 64                               # 每场景目标数
TAU = 0.02                               # 定位命中容差（相对图像边长）
# 单位正方形内一点到其格心的平均距离：E[R] = (s/6)(sqrt(2) + ln(1+sqrt(2)))
E_R_CONST = (np.sqrt(2.0) + np.log(1.0 + np.sqrt(2.0))) / 6.0
E_R2_CONST = 1.0 / np.sqrt(6.0)           # RMS / s


def grid_of(budget):
    """token 预算 -> 网格边长数 G。"""
    return max(1, int(round(budget ** 0.5)))


def encode(pts, G):
    """把 [0,1]^2 内的点编码成"格心坐标"。返回 (格心, 格索引)。"""
    idx = np.minimum((pts * G).astype(int), G - 1)
    centers = (idx + 0.5) / G
    return centers, idx


def scene_stats(pts, centers, idx, G):
    """描述类三项：计数、主导象限、质心。"""
    occ = np.zeros((G, G), dtype=int)
    np.add.at(occ, (idx[:, 0], idx[:, 1]), 1)
    count_est = int(occ.sum())
    h = max(1, G // 2)
    quads = [occ[:h, :h].sum(), occ[:h, h:].sum(),
             occ[h:, :h].sum(), occ[h:, h:].sum()]
    dom_est = int(np.argmax(quads))
    dom_true = int(np.argmax([
        ((pts[:, 0] < 0.5) & (pts[:, 1] < 0.5)).sum(),
        ((pts[:, 0] < 0.5) & (pts[:, 1] >= 0.5)).sum(),
        ((pts[:, 0] >= 0.5) & (pts[:, 1] < 0.5)).sum(),
        ((pts[:, 0] >= 0.5) & (pts[:, 1] >= 0.5)).sum(),
    ]))
    cen_err = float(np.linalg.norm(centers.mean(axis=0) - pts.mean(axis=0)))
    return count_est, dom_est, dom_true, cen_err


def pair_accuracy(pts, centers, rng):
    """相邻目标对"谁在左边"：比格心的 x 列；同列时模型没有信息，按 0.5 猜。

    只用到 x 一个坐标，所以失手的条件是"落在同一列"，概率是 1/G 而不是 1/B。
    """
    a, b = pts[:-1], pts[1:]
    ca, cb = centers[:-1], centers[1:]
    truth = a[:, 0] < b[:, 0]
    same_col = ca[:, 0] == cb[:, 0]
    guess = rng.random(len(truth)) < 0.5
    pred = np.where(same_col, guess, ca[:, 0] < cb[:, 0])
    return float((pred == truth).mean()), float(same_col.mean())


def quant_error(pts, centers, tau):
    """定位误差：均值、RMS、容差命中率。"""
    r = np.linalg.norm(pts - centers, axis=1)
    return float(r.mean()), float(np.sqrt((r ** 2).mean())), float((r <= tau).mean())


def run_uniform():
    rng = np.random.default_rng(SEED)
    rows = []
    for B in BUDGETS:
        G = grid_of(B)
        accs, sames, ms, rmss, hits = [], [], [], [], []
        cnt_ok, dom_ok, cen_err = [], [], []
        for _ in range(N_SCENE):
            pts = rng.random((N_OBJ, 2))
            centers, idx = encode(pts, G)
            c_est, d_est, d_true, cerr = scene_stats(pts, centers, idx, G)
            cnt_ok.append(c_est == N_OBJ)
            dom_ok.append(d_est == d_true)
            cen_err.append(cerr)
            a, same = pair_accuracy(pts, centers, rng)
            m, r_, h = quant_error(pts, centers, TAU)
            accs.append(a); sames.append(same)
            ms.append(m); rmss.append(r_); hits.append(h)
        s = 1.0 / G
        rows.append(dict(
            B=B, G=G, s=s,
            cnt=float(np.mean(cnt_ok)), dom=float(np.mean(dom_ok)),
            cen=float(np.mean(cen_err)),
            pair=float(np.mean(accs)), same=float(np.mean(sames)),
            mean=float(np.mean(ms)), rms=float(np.mean(rmss)),
            hit=float(np.mean(hits)),
            mean_th=E_R_CONST * s, rms_th=E_R2_CONST * s,
            pair_err_th=0.5 / G, cen_th=E_R_CONST * s / np.sqrt(N_OBJ),
            hit_th=min(1.0, np.pi * TAU ** 2 / s ** 2),
        ))
    return rows


def fit_slope(Bs, ys):
    """log-log 斜率：y ~ B^slope。"""
    return float(np.polyfit(np.log(Bs), np.log(ys), 1)[0])


def quadrant_errors(pts, Gq):
    """按象限编码，返回每个点在其象限内的定位误差（象限边长 0.5，格边长 = 0.5/Gq）。"""
    q = (pts[:, 0] >= 0.5).astype(int) * 2 + (pts[:, 1] >= 0.5).astype(int)
    err = np.zeros(len(pts))
    for k in range(4):
        m = q == k
        if not m.any():
            continue
        o = np.array([0.5 * (k // 2), 0.5 * (k % 2)])
        loc, _ = encode((pts[m] - o) * 2.0, Gq)     # 放大到单位方格再编码
        err[m] = np.linalg.norm(pts[m] - (loc / 2.0 + o), axis=1)
    return err, q


def run_allocated(B=64, G_roi=6, G_oth=3):
    """非均匀分配：目标象限 6x6=36 个 token，其余三区各 3x3=9 个（合计 63）。

    均匀对照用同一套代码：整个图 8x8=64 个 token，等价于每个象限 4x4。
    """
    rng = np.random.default_rng(SEED + 777)
    G_uni = grid_of(B)                      # 全图 8x8 -> 每象限 4x4
    e_roi, e_oth, e_uni = [], [], []
    for _ in range(N_SCENE):
        pts = rng.random((N_OBJ, 2))
        _, q = quadrant_errors(pts, G_uni // 2)
        roi = q == 0
        r_roi, _ = quadrant_errors(pts[roi], G_roi)
        r_oth, _ = quadrant_errors(pts[~roi], G_oth)
        r_uni, _ = quadrant_errors(pts, G_uni // 2)
        e_roi.append(r_roi.mean()); e_oth.append(r_oth.mean())
        e_uni.append(r_uni.mean())
    w = 0.25
    roi, oth, uni = (float(np.mean(e_roi)), float(np.mean(e_oth)),
                     float(np.mean(e_uni)))
    return dict(B=B, G_uni=G_uni, G_roi=G_roi, G_oth=G_oth,
                n_uni=G_uni ** 2, n_roi=G_roi ** 2, n_oth=G_oth ** 2,
                s_uni=0.5 / (G_uni // 2), s_roi=0.5 / G_roi, s_oth=0.5 / G_oth,
                roi=roi, oth=oth, uni=uni,
                roi_th=E_R_CONST * 0.5 / G_roi, oth_th=E_R_CONST * 0.5 / G_oth,
                uni_th=E_R_CONST * 0.5 / (G_uni // 2),
                overall=w * roi + (1 - w) * oth)


def main():
    font = plotting.use_chinese_font()
    print("=" * 78)
    print("Demo W - 视觉 token 预算：描述不怕压，定位怕")
    print("=" * 78)
    print(f"字体: {font or '未找到 CJK 字体'}")
    print(f"每场景 {N_OBJ} 个目标，{N_SCENE} 个场景 | 容差 tau = {TAU}")
    print("编码器 = G x G 均匀网格（G = round(sqrt(B))），每个 token 只携带格心坐标。")
    print("不训练任何模型：格子边长 s = 1/G 就是全部假设。")

    rows = run_uniform()
    Bs = [r['B'] for r in rows]

    print("\n" + "-" * 78)
    print("① 整数统计：精确不变；聚合统计：按 sqrt(N) 缩水")
    print("-" * 78)
    print(f"{'token B':>9}{'G':>5}{'计数正确率':>13}{'主导象限准确率':>17}"
          f"{'质心误差':>12}{'逐目标平均误差':>17}{'倍数':>9}")
    for r in rows:
        print(f"{r['B']:>9}{r['G']:>5}{r['cnt']:>13.4f}{r['dom']:>17.4f}"
              f"{r['cen']:>12.2e}{r['mean']:>17.4f}{r['mean'] / r['cen']:>9.1f}")
    print(f"   -> 计数与主导象限在所有档位都是 1.0000：它们是**格占用数的整数求和**，")
    print(f"      切成几格不改变求和结果，所以对 token 预算完全免疫。")
    print(f"   -> 质心误差随预算涨，但始终比逐目标误差低 sqrt({N_OBJ}) = "
          f"{np.sqrt(N_OBJ):.1f} 这一档（实测倍数 {np.mean([r['mean'] / r['cen'] for r in rows]):.1f}）：")
    print(f"      逐目标量化误差是零均值的，N 个一平均就下去了。这就是「聚合」的防御力。")

    print("\n" + "-" * 78)
    print("② 读坐标的判读：左右（用 1 个坐标）失手率 = 0.5/G")
    print("-" * 78)
    print(f"{'token B':>9}{'同列率(实测)':>15}{'1/G':>10}{'左右判断准确率':>17}"
          f"{'闭式 1-0.5/G':>15}")
    for r in rows:
        print(f"{r['B']:>9}{r['same']:>15.4f}{1.0 / r['G']:>10.4f}"
              f"{r['pair']:>17.4f}{1.0 - r['pair_err_th']:>15.4f}")
    print(f"   -> 同列率实测贴 1/G，**不是 1/B**：判断左右只需要一个坐标列，两个点")
    print(f"      落进同一列的概率是 1/G = 1/sqrt(B)。少用一个坐标，退化就慢一半。")

    print("\n" + "-" * 78)
    print("③ 单体定位：误差就是量化误差，均值 0.3826*s，指数 -1/2")
    print("-" * 78)
    print(f"{'token B':>9}{'边长 s':>10}{'平均误差':>11}{'闭式 0.3826s':>15}"
          f"{'RMS':>10}{'闭式 s/sqrt6':>14}")
    for r in rows:
        print(f"{r['B']:>9}{r['s']:>10.4f}{r['mean']:>11.4f}{r['mean_th']:>15.4f}"
              f"{r['rms']:>10.4f}{r['rms_th']:>14.4f}")
    print(f"   -> 实测贴着闭式。预算除以 144（B 从 {BUDGETS[0]} 到 {BUDGETS[-1]}），"
          f"平均误差涨 {rows[-1]['mean'] / rows[0]['mean']:.1f} 倍")
    print(f"      （= G 之比 {rows[0]['G'] / rows[-1]['G']:.0f}）；同一区间里左右判断的失手率也涨 "
          f"{(0.5 / rows[-1]['G']) / (0.5 / rows[0]['G']):.0f} 倍。")
    print(f"   -> 两条曲线斜率相同（拟合 log-log 斜率：定位 "
          f"{fit_slope(Bs, [r['mean'] for r in rows]):+.3f}，左右 "
          f"{fit_slope(Bs, [r['pair_err_th'] for r in rows]):+.3f}），差的是常数"
          f"（0.3826 对 0.5）。")
    print("      所以「还能说谁在左边」和「还能指出在哪」是一条线，不是两条。")

    print("\n" + "-" * 78)
    print("④ 容差：命中率与 B 成正比，松容差会掩盖一切")
    print("-" * 78)
    print(f"{'token B':>9}{'边长 s':>10}{'命中率(实测)':>15}{'闭式 pi*tau^2/s^2':>19}"
          f"{'B 之比':>9}{'命中率之比':>12}")
    prev, prev_b = None, None
    for r in rows:
        g = "-" if prev is None else f"{prev / r['hit']:.2f}x"
        b = "-" if prev_b is None else f"{prev_b / r['B']:.2f}x"
        print(f"{r['B']:>9}{r['s']:>10.4f}{r['hit']:>15.4f}{r['hit_th']:>19.4f}"
              f"{b:>9}{g:>12}")
        prev, prev_b = r['hit'], r['B']
    print(f"   -> 命中率与 B 成正比：逐档的「B 之比」与「命中率之比」逐行吻合"
          f"（B 从 {BUDGETS[2]} 到 {BUDGETS[0]} 是 9 倍，")
    print(f"      命中率从 {rows[2]['hit']:.4f} 涨到 {rows[0]['hit']:.4f}，"
          f"{rows[0]['hit'] / rows[2]['hit']:.1f} 倍）。")
    print("      容差一松，所有预算看起来都能用——那是任务本身不需要细粒度，不是模型变好了。")

    print("\n" + "-" * 78)
    print("⑤ 分配方向：把 token 集中给目标象限，该区更好、整体更差")
    print("-" * 78)
    a = run_allocated()
    print(f"预算固定 B={a['B']}。均匀 = {a['G_uni']}x{a['G_uni']} = {a['n_uni']} 个 token，"
          f"每象限 {a['G_uni'] // 2}x{a['G_uni'] // 2}")
    print(f"              集中 = 目标象限 {a['G_roi']}x{a['G_roi']}={a['n_roi']} 个，"
          f"其余三区各 {a['G_oth']}x{a['G_oth']}={a['n_oth']} 个，合计 "
          f"{a['n_roi'] + 3 * a['n_oth']} 个")
    print(f"{'':>16}{'格边长 s':>11}{'平均定位误差':>15}{'闭式':>10}{'误差倍数':>12}")
    for name, s_, e_, t_ in [("均匀-全图", a['s_uni'], a['uni'], a['uni_th']),
                             ("集中-目标区", a['s_roi'], a['roi'], a['roi_th']),
                             ("集中-其余三区", a['s_oth'], a['oth'], a['oth_th'])]:
        print(f"{name:>16}{s_:>11.4f}{e_:>15.4f}{t_:>10.4f}{e_ / a['uni']:>12.3f}")
    print(f"{'集中-整体平均':>16}{'-':>11}{a['overall']:>15.4f}{'-':>10}"
          f"{a['overall'] / a['uni']:>12.3f}")
    print(f"   -> 目标象限好 {a['uni'] / a['roi']:.2f} 倍，代价是其余三区各差 "
          f"{a['oth'] / a['uni']:.2f} 倍；按面积加权，整体平均反而差 "
          f"{a['overall'] / a['uni']:.2f} 倍。token 分配是有方向的取舍。")

    print("\n" + "-" * 78)
    print("外推边界（必须与结论一起读）")
    print("-" * 78)
    print("   本实验把视觉编码器建模成**均匀网格 + 格心坐标**，只保留分辨率这一个变量，")
    print("   量的是「格子边长 s 与三类处理方式的关系」，不是某个真实 VLM 的分数。")
    print("   已知不能外推的部分：(a) 真实 token 不是格心而是学出来的特征，相邻格有")
    print("   重叠感受野，定位误差会小于这里的量化误差；(b) 真实模型的描述类任务也会随")
    print("   预算退化（注意力被稀释、语言先验接管），这里让它们精确不变，是**上界**；")
    print("   (c) 真实基准的容差由任务定义，不是这里设的 0.02。")

    # ---- 出图 ----
    fig, axes = plotting.plt.subplots(1, 3, figsize=(15.2, 4.6))
    c1, c2, c3 = "#4C72B0", "#C44E52", "#55A868"

    ax = axes[0]
    ax.plot(Bs, [r['mean'] for r in rows], "o-", color=c2, label="实测平均误差")
    ax.plot(Bs, [r['mean_th'] for r in rows], "x--", color=c2, alpha=0.6,
            label="闭式 0.3826s")
    ax.plot(Bs, [r['cen'] for r in rows], "D-", color=c3, label="质心（聚合）")
    ax.set_xlabel("视觉 token 数 B")
    ax.set_ylabel("误差（相对边长，对数）")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8, loc="center left")
    ax.set_title("(1) 定位与聚合：同一斜率，差 sqrt(N)")

    ax = axes[1]
    ax.plot(Bs, [1.0 - r['pair'] for r in rows], "^-", color=c1,
            label=f"左右判断失手率（斜率 {fit_slope(Bs, [1.0 - r['pair'] for r in rows]):+.2f}）")
    ax.plot(Bs, [r['mean'] for r in rows], "s-", color=c2,
            label=f"定位误差（斜率 {fit_slope(Bs, [r['mean'] for r in rows]):+.2f}）")
    ax.plot(Bs, [r['hit'] for r in rows], "o:", color="#8C8C8C",
            label="容差命中率 tau=0.02")
    ax.set_xlabel("视觉 token 数 B")
    ax.set_ylabel("对数轴")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7.5, loc="center left")
    ax.set_title("(2) 两条平行线：读坐标的都按 1/sqrt(B)")

    ax = axes[2]
    labels = ["均匀\n每区 %d" % (a['G_uni'] // 2) ** 2,
              "目标区\n%d" % a['n_roi'], "其余三区\n%d" % a['n_oth']]
    vals = [a['uni'], a['roi'], a['oth']]
    ax.bar(range(3), vals, color=["#8C8C8C", c3, c2])
    for i, v in enumerate(vals):
        ax.text(i, v * 1.02, f"{v:.4f}", ha="center", fontsize=8)
    ax.axhline(a['uni'], color="#8C8C8C", ls="--", lw=1, alpha=0.8)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("定位平均误差")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"(3) 集中分配 B={a['B']}：整体 {a['overall'] / a['uni']:.2f}x")

    fig.suptitle("token 预算是分辨率的代理量：整数统计不怕压，读坐标怕", y=1.0)
    fig.tight_layout()
    plotting.save(fig, "w_token_budget.png")


if __name__ == "__main__":
    main()
