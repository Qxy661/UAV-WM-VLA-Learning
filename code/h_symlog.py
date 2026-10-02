"""
Demo H — symlog 把"尺度"统一到什么程度？

对应文档：docs/02-世界模型专题/01-世界模型发展史.md

要验证的结论：4.3 节说 DreamerV3 靠 symlog 变换统一不同尺度的信号，
从而做到单一算法、固定超参数跨域泛化。"统一"这个词是可以量的：
同一个网络、同一个学习率、同一份数据，目标值跨三个数量级时，
把损失算在原始空间和算在 symlog 空间，各尺度上的相对误差差多少。

做法是把回归目标按尺度分成四组（×1 / ×10 / ×100 / ×1000），
两次训练逐行对称，唯一变量是损失算在哪个空间。

关键设计
  - 输入里带 one-hot 的尺度组标识，网络有足够信息判断量级，
    学不好只可能是优化问题，不是容量问题
  - 评估一律在原始空间做，symlog 那组要先 symexp 还原再算误差
  - 同时报"总体 MSE"和"分组相对误差"：前者会给出误导性的乐观结论

运行：py -3.9 code/h_symlog.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import plotting

SEED = 0
N_TRAIN, N_TEST = 4096, 2048
SCALES = (1.0, 10.0, 100.0, 1000.0)     # 四组尺度，跨三个数量级
HIDDEN, STEPS, LR = 64, 3000, 1e-3


def symlog(x):
    """f(x) = sign(x) * log(1 + |x|)，把任意尺度压到对称的有界区间。"""
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x):
    """symlog 的逆。取绝对值再指数，符号单独还原。"""
    return torch.sign(x) * torch.expm1(x.abs())


class MLP(nn.Module):
    """6 维输入 -> 1 维输出。两个隐藏层，两次训练用的是同一个结构。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, HIDDEN), nn.ReLU(),
                                 nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                                 nn.Linear(HIDDEN, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_data(n, seed):
    """u 决定幅度与符号，k 决定尺度组。x 里两种信息都给全。"""
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(n, generator=g) * 2 - 1                      # [-1, 1]
    k = torch.randint(0, len(SCALES), (n,), generator=g)
    onehot = F.one_hot(k, len(SCALES)).float()
    x = torch.cat([u[:, None], torch.sin(4 * u)[:, None], onehot], -1)
    y = torch.tensor(SCALES)[k] * (u ** 3 + 0.5 * torch.sin(3 * u))   # 重尾：值域随组差千倍
    return x, y, k


def train(mode, x, y, x_eval):
    """mode='raw' 直接回归 y；mode='symlog' 回归 symlog(y)，损失也在 symlog 空间算。
    评估统一在原始空间：symlog 那组的输出要先 symexp 还原再比。"""
    torch.manual_seed(SEED)                                     # 两次训练起点完全相同
    net = MLP()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    target = y if mode == "raw" else symlog(y)
    for _ in range(STEPS):
        loss = F.mse_loss(net(x), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = net(x_eval)
    return (pred if mode == "raw" else symexp(pred)).clamp(-3e3, 3e3)


def report(name, pred, y, k):
    """分组报相对误差。分母用 |y| 的下限兜底，免得 u 落在 0 附近把比值放大。"""
    print(f"\n{name}")
    print(f"  {'尺度组':<8}{'相对误差':>10}{'该组 MSE':>12}")
    rel_all = []
    for i, s in enumerate(SCALES):
        m = k == i
        rel = ((pred[m] - y[m]).abs() / y[m].abs().clamp(min=0.05 * s)).mean().item()
        mse = F.mse_loss(pred[m], y[m]).item()
        rel_all.append(rel)
        print(f"  x{s:<7g}{rel:>9.1%}{mse:>12.1f}")
    overall = F.mse_loss(pred, y).item()
    worst_ratio = max(rel_all) / max(min(rel_all), 1e-9)
    print(f"  总体 MSE {overall:>10.1f}    最差组/最好组 {worst_ratio:>6.1f}x")
    return rel_all, overall


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    font = plotting.use_chinese_font()
    print(f"字体: {font or '未找到 CJK 字体，图中中文可能显示为方块'}\n")

    xtr, ytr, ktr = make_data(N_TRAIN, SEED)
    xte, yte, kte = make_data(N_TEST, SEED + 1)
    print(f"训练集 {N_TRAIN} 条，测试集 {N_TEST} 条，四组尺度 {SCALES}")
    med = [ytr[ktr == i].abs().median().item() for i in range(len(SCALES))]
    print("各组 |y| 的中位数 " + " / ".join(f"{m:.1f}" for m in med) + "（相邻组差 10 倍）")

    res = {}
    for mode, title in (("raw", "原始空间 MSE（不加 symlog）"), ("symlog", "symlog 空间 MSE")):
        res[mode] = train(mode, xtr, ytr, xte)
        report(title, res[mode], yte, kte)

    print("\n结论：不加 symlog 时总体 MSE 看着更小，因为梯度被 x1000 那组独占；"
          "\n      代价是小尺度组几乎没被学到。symlog 把四组的相对误差拉到一个量级上，"
          "\n      这才是固定超参跨域能成立的前提。")

    # ---- 出图 ----
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = {"raw": "#c0392b", "symlog": "#2471a3"}

    ax = axes[0]
    w = 0.36
    xs = torch.arange(len(SCALES))
    for j, (mode, lab) in enumerate((("raw", "原始 MSE"), ("symlog", "symlog"))):
        p = res[mode]
        rel = [((p[kte == i] - yte[kte == i]).abs()
                / yte[kte == i].abs().clamp(min=0.05 * s)).mean().item()
               for i, s in enumerate(SCALES)]
        ax.bar(xs + (j - 0.5) * w, rel, w, label=lab, color=colors[mode])
        for xi, r in zip(xs + (j - 0.5) * w, rel):
            ax.text(xi, r * 1.15, f"{r:.0%}", ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(xs); ax.set_xticklabels([f"x{s:g}" for s in SCALES])
    ax.set_xlabel("目标值的尺度组"); ax.set_ylabel("相对误差（对数轴）")
    ax.set_title("(a) 分组相对误差：symlog 把四组拉平")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    m = kte == 0                                   # 最小的那一组，x1
    lo, hi = -2, 2
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="理想")
    ax.scatter(yte[m], res["raw"][m], s=6, alpha=0.35, color=colors["raw"], label="原始 MSE")
    ax.scatter(yte[m], res["symlog"][m], s=6, alpha=0.35, color=colors["symlog"], label="symlog")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("真值（尺度组 x1）"); ax.set_ylabel("预测值")
    ax.set_title("(b) 最小尺度上的预测：不加 symlog 与真值几乎无关")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)

    fig.suptitle("symlog 统一的不是数值范围，是各尺度上的优化难度", fontsize=13)
    fig.tight_layout()
    plotting.save(fig, "h_symlog.png")


if __name__ == "__main__":
    main()
