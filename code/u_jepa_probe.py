"""
Demo U — 重建式表征 vs 预测式表征：瓶颈位数一样，留住的信号不一样

对应文档：docs/02-世界模型专题/08-联合嵌入预测与潜空间世界模型.md

要验证的结论：在同一个固定宽度的瓶颈下，"把输入重建出来"与"预测下一步"这两个
目标会**选出不同的方向**。重建按方差排序，于是把高方差但不可预测、与任务无关的
方向装进码字；预测按"下一步能解释多少"排序，于是优先装进动作能推动、时间上连贯的
方向。结果：预测式码字的下游线性探针更准，而重建误差更差——两个目标各自最优，
选哪个取决于下游要什么。

**这是一个线性-高斯最小模型，不是 JEPA 本身。**见文末"外推边界"。

装置（纯解析，闭式解，不训练任何模型）：
  潜变量 z = s + eta，8 维，分两组——
    coherent（4 维）时间连贯、动作可推：a = 0.95，平稳方差 1.0
    nuisance（4 维）时间近白、与任务无关：a = 0.30，平稳方差 2.0
  观测 x = R z，R 是固定的随机正交阵（只为让方向不与坐标轴对齐，两个判据都
  在特征分解后求解，与 R 无关）
  观测噪声 eta ~ N(0, sigma^2 I)，sigma^2 是自变量
  下游任务 y = c.s，c 只在 coherent 组上非零——即任务只关心连贯结构

  两个编码器都是 8x6 的正交列投影，各自按自己的判据在闭式下最优：
    重建派 = Cov(x) 的前 6 个特征方向（最小重建误差）
    预测派 = 最大化"下一步能被线性解释的方差"的前 6 个方向

三条诊断量：
  探针 R2（闭式）  无限数据下最优线性探针的解释力——只反映"码字里有几个信号方向"
  探针 R2（实测）  N_train = 100 的 ridge 探针，多次随机划分取均值——额外反映
                   "码字里带了几个高方差的无用方向"
  重建 RMSE         sqrt(tr(Cov(x)(I - P)) / d)，P 是编码器的投影

自检点：
  (1) 重建派选中的 6 个方向里，落在无关组的能量应占 4/6；预测派是 2/6。
  (2) 重建派的 RMSE 应恒不高于预测派（它就是这个指标下的最优解）——手算
      sigma^2 = 0 时为 sqrt(1/8) = 0.354 与 sqrt(2.5/8) = 0.559。
  (3) 实测与同口径闭式 R2 的差应是几个百分点量级（有限样本的估计方差），
      且两派的排序在有限样本下不翻转。

运行：py -3.9 code/u_jepa_probe.py
纯 CPU，约 10 秒；不训练任何模型。
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import plotting

SEED = 11
D = 8                       # 潜变量维数
N_COH = 4                   # 前 4 维：时间连贯、任务相关
N_NUI = 4                   # 后 4 维：时间近白、任务无关
# 每一步的平稳方差与时间系数**逐维取不同值**。这是必须的：若组内取相同值，
# 协方差矩阵就有 4 重简并特征值，eigh 返回的组内基是任意的（LAPACK 的实现细节），
# 而被保留的那个子空间直接决定探针精度——结论会由 LAPACK 决定，不由模型决定。
LAM_COH = [1.00, 0.80, 0.60, 0.40]
A_COH = [0.95, 0.93, 0.91, 0.89]
LAM_NUI = [2.00, 1.70, 1.40, 1.10]
A_NUI = [0.30, 0.27, 0.24, 0.21]
RANK = 6                    # 瓶颈宽度（两个编码器共用，这是"公平比较"的前提）
SIGMA2 = [0.01, 0.1, 0.5, 1.0, 3.0]    # 观测噪声方差，自变量
N_SEQ, T_SEQ, BURN = 6, 3000, 100
N_TRAIN, N_TEST, N_SPLIT = 150, 4000, 200
RIDGE_ALPHA = 1e-3


def build_world():
    """返回 (R, a, lam)：观测正交阵、各方向的时间系数、各方向的平稳方差。

    A = diag(a) 是对角的，所以两个判据的特征分解都在潜坐标下做，再乘 R 回到观测坐标。
    """
    rng = np.random.default_rng(SEED)
    R, _ = np.linalg.qr(rng.standard_normal((D, D)))       # 固定随机正交阵
    a = np.array(A_COH + A_NUI)
    lam = np.array(LAM_COH + LAM_NUI)
    # 平稳要 (1-a^2)*lam 的过程噪声；这里直接给平稳协方差，不需要模拟预热
    return R, a, lam


def label_vec(lam):
    """任务只关心连贯组。c 在每组内按 1/sqrt(lam_i) 加权，使每个方向对 var(y) 等贡献。"""
    c = np.zeros(D)
    c[:N_COH] = 0.5 / np.sqrt(lam[:N_COH])     # var(y) = sum c_i^2 lam_i = 4 * 0.25 = 1
    return c


def gen_data(R, a, lam, sigma2, rng):
    """模拟一条长轨迹，返回 (X, Y)。X = R(s+eta)，Y = c.s。"""
    s = np.zeros((T_SEQ, D))
    # 平稳初值：每个方向的标准差是 sqrt(lam_i)
    s[0] = np.sqrt(lam) * rng.standard_normal(D)
    for t in range(1, T_SEQ):
        s[t] = a * s[t - 1] + np.sqrt((1 - a ** 2) * lam) * rng.standard_normal(D)
    z = s + np.sqrt(sigma2) * rng.standard_normal((T_SEQ, D))
    X = (R @ z.T).T
    Y = s @ label_vec(lam)
    sl = slice(BURN, None)
    return X[sl], Y[sl]


def encoders(R, a, lam, sigma2):
    """返回两个编码器（8xRANK，正交列）与投影矩阵。

    重建派：Cov(x) = R diag(lam + sigma2) R^T 的前 RANK 个特征方向。
    预测派：最大化 (v^T C v)^2 / (v^T D v)——即方向 v 上"下一步能被线性解释的
            方差"。等价于对 M = D^{-1/2} C D^{-1/2} 做特征分解，M 是对角的，
            所以取 |M_ii| 最大的 RANK 个方向。

    两个判据都在潜坐标下对角化，逐维参数互不相同（见常量处的注释），所以最优的
    RANK 维子空间唯一，不依赖 eigh 在简并子空间里的取基方式。
    """
    D_cov = np.diag(lam + sigma2)
    C_cross = np.diag(a * lam)
    Sig_x = R @ D_cov @ R.T

    w, V = np.linalg.eigh(Sig_x)
    E_rec = V[:, np.argsort(w)[::-1][:RANK]]

    M = np.diag(1.0 / np.sqrt(np.diag(D_cov))) @ C_cross @ np.diag(
        1.0 / np.sqrt(np.diag(D_cov)))
    m, Vm = np.linalg.eigh(M)
    # 还原坐标：令 w = D^{1/2} v，判据化成 (w^T M w)^2 / ||w||^2，
    # 极值点在 M 的特征向量上；回到原坐标要再乘 D^{-1/2} 并归一化。
    Vp = np.diag(1.0 / np.sqrt(np.diag(D_cov))) @ Vm
    Vp /= np.linalg.norm(Vp, axis=0, keepdims=True)
    E_pred = R @ Vp[:, np.argsort(np.abs(m))[::-1][:RANK]]
    return E_rec, E_pred


def group_energy(E, R):
    """编码器各方向在 coherent / nuisance 两组上占的能量比。"""
    V = R.T @ E                       # 回到潜坐标
    e_coh = (V[:N_COH] ** 2).sum()
    e_nui = (V[N_COH:] ** 2).sum()
    tot = e_coh + e_nui
    return e_coh / tot, e_nui / tot


def recon_rmse(E, R, lam, sigma2):
    """sqrt(tr(Cov(x)(I - P)) / d)，P = E E^T。"""
    D_cov = np.diag(lam + sigma2)
    kept = np.trace(E.T @ R @ D_cov @ R.T @ E)
    return np.sqrt((np.trace(D_cov) - kept) / D)


def pop_r2(E, R, a, lam, sigma2):
    """无限数据下最优线性探针的解释力（闭式）。

    y = c.s，cov(h, y) = E^T R Sigma_s c，cov(h) = E^T Cov(x) E。
    """
    Sig_s = np.diag(lam)
    c = label_vec(lam)
    D_cov = np.diag(lam + sigma2)
    g = E.T @ R @ Sig_s @ c
    H = E.T @ (R @ D_cov @ R.T) @ E
    return float(g @ np.linalg.solve(H, g)) / float(c @ Sig_s @ c)


def ridge_probe(E, Xtr, Ytr, Xte, Yte):
    Htr, Hte = Xtr @ E, Xte @ E
    G = Htr.T @ Htr + RIDGE_ALPHA * np.eye(Htr.shape[1])
    w = np.linalg.solve(G, Htr.T @ Ytr)
    pred = Hte @ w
    return 1.0 - float(((Yte - pred) ** 2).sum()) / float(
        ((Yte - Yte.mean()) ** 2).sum())


def main():
    font = plotting.use_chinese_font()
    print("=" * 78)
    print("Demo U - 重建式表征 vs 预测式表征（线性-高斯最小模型）")
    print("=" * 78)
    print(f"字体: {font or '未找到 CJK 字体'}")
    print(f"潜维数 {D}：连贯 {N_COH} 维 var={LAM_COH} a={A_COH}")
    print(f"          无关 {N_NUI} 维 var={LAM_NUI} a={A_NUI}")
    print(f"瓶颈 {RANK} 维（两派共用）| 探针训练样本 {N_TRAIN}，"
          f"{N_SPLIT} 次随机划分取均值")
    print("下游任务只关心连贯组。不训练任何模型：两个编码器都是闭式解。")

    R, a, lam = build_world()
    c = label_vec(lam)
    print(f"\n任务标签 y = c.s，var(y) = {c @ np.diag(lam) @ c:.4f}"
          f"（c 只在连贯组上非零，组内按 1/sqrt(var) 加权，四维等贡献）")

    print("\n" + "-" * 78)
    print("① 两个编码器选中的方向落在哪一组")
    print("-" * 78)
    print(f"{'sigma^2':>9}{'重建派 连贯/无关':>22}{'预测派 连贯/无关':>22}")
    enc = {}
    for s2 in SIGMA2:
        e_rec, e_pred = encoders(R, a, lam, s2)
        enc[s2] = (e_rec, e_pred)
        cr, nr = group_energy(e_rec, R)
        cp, npp = group_energy(e_pred, R)
        print(f"{s2:>9.3f}{cr:>13.3f}{nr:>9.3f}{cp:>13.3f}{npp:>9.3f}")
    print("   -> 重建派按方差排序，把 4 个高方差的方向全装进来了；预测派只装 2 个，"
          "把另外 4 个位置留给时间连贯的方向。")
    print("   -> 这一栏与 sigma^2 无关：本模型里两个判据的**排序**都不随噪声变，"
          "噪声只改变码字里信号与噪声的相对比重。")

    print("\n" + "-" * 78)
    print("② 重建误差：预测派为它自己的目标付出的代价")
    print("-" * 78)
    print(f"{'sigma^2':>9}{'重建派 RMSE':>16}{'预测派 RMSE':>16}{'相对':>10}")
    for s2 in SIGMA2:
        e_rec, e_pred = enc[s2]
        r1, r2 = recon_rmse(e_rec, R, lam, s2), recon_rmse(e_pred, R, lam, s2)
        print(f"{s2:>9.3f}{r1:>16.4f}{r2:>16.4f}{r2 / r1:>10.3f}")
    e_rec0, e_pred0 = enc[SIGMA2[0]]
    print(f"   -> 重建派的 RMSE 恒不高于预测派（它是这个指标下的最优解）。"
          f"噪声越大，两者的相对差距越小：sigma^2 = {SIGMA2[0]} 时 "
          f"{recon_rmse(e_pred0, R, lam, SIGMA2[0]) / recon_rmse(e_rec0, R, lam, SIGMA2[0]):.2f} 倍，"
          f"sigma^2 = {SIGMA2[-1]} 时 "
          f"{recon_rmse(enc[SIGMA2[-1]][1], R, lam, SIGMA2[-1]) / recon_rmse(enc[SIGMA2[-1]][0], R, lam, SIGMA2[-1]):.2f} 倍。"
          f"因为噪声占比一高，谁也留不住。")

    print("\n" + "-" * 78)
    print("③ 下游线性探针：闭式（无限数据）与实测（少样本）")
    print("-" * 78)
    print(f"{'sigma^2':>9}{'闭式 重建':>12}{'闭式 预测':>12}{'实测 重建':>12}"
          f"{'实测 预测':>12}{'实测 全维':>12}")
    rows = []
    for s2 in SIGMA2:
        e_rec, e_pred = enc[s2]
        rng = np.random.default_rng(SEED + int(s2 * 1000))
        X, Y = gen_data(R, a, lam, s2, rng)
        acc_r, acc_p, acc_f = [], [], []
        E_full = np.eye(D)
        for _ in range(N_SPLIT):
            perm = rng.permutation(X.shape[0])
            itr = perm[:N_TRAIN]
            ite = perm[N_TRAIN:N_TRAIN + N_TEST]
            Xtr, Ytr, Xte, Yte = X[itr], Y[itr], X[ite], Y[ite]
            acc_r.append(ridge_probe(e_rec, Xtr, Ytr, Xte, Yte))
            acc_p.append(ridge_probe(e_pred, Xtr, Ytr, Xte, Yte))
            acc_f.append(ridge_probe(E_full, Xtr, Ytr, Xte, Yte))
        m_r, m_p, m_f = np.mean(acc_r), np.mean(acc_p), np.mean(acc_f)
        rows.append((s2, pop_r2(e_rec, R, a, lam, s2), pop_r2(e_pred, R, a, lam, s2),
                     m_r, m_p, m_f))
        print(f"{s2:>9.3f}{rows[-1][1]:>12.4f}{rows[-1][2]:>12.4f}"
              f"{m_r:>12.4f}{m_p:>12.4f}{m_f:>12.4f}")
    print("   -> 闭式一栏只反映「码字里有几个信号方向」：预测派 4 个全在，重建派只有 2 个。")
    print("   -> 实测与闭式的差值就是有限样本的估计方差，量级是几个百分点，"
          "方向不固定（有的档实测反而更高，因为 ridge 收缩本身也能帮上忙）。"
          "**这个最小模型量不出「无用方向额外拖累探针」这条更细的说法**："
          "要把那一条量出来，得把训练样本压到与瓶颈宽度同量级，"
          "那时测的是估计噪声，不是结论。")
    print("   -> 上界（不压瓶颈）与预测派几乎重合：说明在 6 维瓶颈下的损失，"
          "主要来自「丢掉了几个信号方向」，而不是「多装了无用方向」。")
    print("   -> 两栏都不是「预测派一定更好」：把下游换成重建、或让任务落在近白组上，"
          "排序会反过来——判据要跟着下游定。")

    print("\n" + "-" * 78)
    print("外推边界（必须与结论一起读）")
    print("-" * 78)
    print("   本实验是**线性-高斯最小模型**：编码器是线性正交投影，动力学是线性的，")
    print("   噪声是高斯的，两个判据都有闭式解。它量的是**两个目标函数的分歧**，")
    print("   不是某个真实 JEPA 或某个真实自编码器的性能。")
    print("   已知不能外推的部分：真实 JEPA 的编码器是非线性的，表征塌缩要靠专门的")
    print("   正则项（不是这里的高斯噪声）；真实世界的时间连贯性不会像这里一样")
    print("   恰好分成两组常数。把结论外推到真实模型规模，需要另做实验。")

    # ---- 出图 ----
    fig, axes = plotting.plt.subplots(1, 3, figsize=(15.2, 4.6))
    s2s = [r[0] for r in rows]
    col_rec, col_pred, col_full = "#C44E52", "#4C72B0", "#55A868"

    ax = axes[0]
    ax.plot(s2s, [r[1] for r in rows], "o--", color=col_rec, label="重建派（闭式）")
    ax.plot(s2s, [r[2] for r in rows], "o-", color=col_pred, label="预测派（闭式）")
    ax.plot(s2s, [r[3] for r in rows], "s--", color=col_rec, alpha=0.55,
            label="重建派（实测）")
    ax.plot(s2s, [r[4] for r in rows], "s-", color=col_pred, alpha=0.55,
            label="预测派（实测）")
    ax.plot(s2s, [r[5] for r in rows], "^:", color=col_full,
            label="不压瓶颈（实测上界）")
    ax.set_xlabel("观测噪声方差 sigma^2")
    ax.set_ylabel("下游线性探针 R2")
    ax.set_title("① 下游探针：预测派全面更高（有限样本下排序不翻转）")
    ax.set_xscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)

    ax = axes[1]
    ax.plot(s2s, [recon_rmse(enc[s][0], R, lam, s) for s in s2s], "o-",
            color=col_rec, label="重建派")
    ax.plot(s2s, [recon_rmse(enc[s][1], R, lam, s) for s in s2s], "s-",
            color=col_pred, label="预测派")
    ax.set_xlabel("观测噪声方差 sigma^2")
    ax.set_ylabel("重建 RMSE (m)")
    ax.set_title("② 重建误差：预测派为它自己的目标付代价")
    ax.set_xscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = axes[2]
    s2 = SIGMA2[2]
    e_rec, e_pred = enc[s2]
    cr, nr = group_energy(e_rec, R)
    cp, npp = group_energy(e_pred, R)
    xs = np.arange(2)
    ax.bar(xs - 0.18, [cr, nr], width=0.36, color=col_rec, label="重建派")
    ax.bar(xs + 0.18, [cp, npp], width=0.36, color=col_pred, label="预测派")
    ax.set_xticks(xs)
    ax.set_xticklabels(["时间连贯\n（任务相关）", "时间近白\n（任务无关）"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel("编码器方向的能量占比")
    ax.set_title(f"③ 瓶颈装了什么（sigma^2 = {s2}）")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    fig.suptitle("同一个瓶颈，两个目标选出两组方向", y=1.0)
    fig.tight_layout()
    plotting.save(fig, "wm8_jepa_probe.png")


if __name__ == "__main__":
    main()
