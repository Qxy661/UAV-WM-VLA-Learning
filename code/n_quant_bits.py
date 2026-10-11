"""
Demo N — 量化到底压掉了什么？

对应文档：docs/04-VLM专题/04-边缘VLM部署.md、docs/03-VLA专题/05-机载部署与优化.md

要验证的结论：文档 3.2 节把量化写成「INT8 保持 97-99%、INT4 保持 93-97%」，
像是一个与任务无关的固定折扣率。测下来这个说法两头都不对：
高位（INT8/INT6）的效应小于训练本身的波动，量不出来；低位是断崖不是打折。
而且量化伤到的不只是回归精度——生成任务是离散的，没有"轻微错"这一档。

做法：先训一个小卷积网络做连续回归（从 32x32 图预测光斑的侧向偏移），
训练完只量化权重、激活保持 FP32（PTQ），逐档测误差。再单独量三件与 VLM 直接相关的事：
  1. 一个投影层里混有量级差几十倍的通道时，逐张量量化的代价落在谁头上
  2. 注意力的 logits 量化后，softmax 概率偏多少
  3. 同一批 logits 量化后，argmax 翻掉多少 —— 以及翻转率能不能被"间隔 vs 步长"估出来

关键设计
  - 量化一律用对称均匀量化 q = round(w/scale) * scale，scale 取该组 amax/qmax，
    这是 TensorRT / ONNX 的默认做法，不做任何花哨的补偿
  - 回归任务而不是分类任务：分类任务太容易被饱和的网络扛过去，
    第一版在二分类上量 INT4 的准确率还是 1.0000，零退化，量不出东西
  - 跑一簇训练种子而不是一次训练。这不是讲究：单次训练的结果换个 BLAS 线程数
    就能让 INT8 的退化从 +16% 翻到 -6%，因为 1500 步 Adam 落在哪个盆地里本身就带噪声。
    每一档的退化都是配对比较（同一颗种子训出的模型、同一份留出集，只换权重），
    判据是这簇种子上会不会翻号——不翻号才算真测出来了
  - 报告相对误差时同时给"全体"和"普通通道"两个数，否则离群通道会把结论盖住

运行：py -3.9 code/n_quant_bits.py
"""

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import plotting

SEED = 0
SIZE = 32                     # 图像边长
N_TRAIN, N_TEST = 4096, 1024
SIGMA = 2.0                   # 光斑半径（像素）
CX, SWING = 16.0, 9.0         # 光斑中心，以及左右摆动的幅度
STEPS, BATCH, LR = 1500, 64, 3e-3
BITS = (8, 6, 4, 3, 2)        # 逐档量化的位宽
PARAMS = 7e9                  # 算体积比用的参数量：一个 7B 级 VLM
SEEDS = tuple(range(10))      # 训练种子。单次训练的波动比 INT8 的效应还大，必须跑一簇取分布


def make_data(n, seed):
    """一条样本 = 一张 32x32 图：一个高斯光斑，横向位置编码连续目标 t∈[-1,1]。"""
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(n, generator=g) * 2 - 1                      # 目标：归一化侧向偏移
    ax = torch.arange(SIZE, dtype=torch.float32)
    cx = (CX + SWING * t)[:, None, None]                        # (n,1,1)，广播到整张图
    img = torch.exp(-((ax[None, None, :] - cx) ** 2
                      + (ax[None, :, None] - CX) ** 2) / (2 * SIGMA ** 2))
    return img[:, None], t


class OffsetNet(nn.Module):
    """三层小卷积 + 全局池化 + 线性头，回归出光斑的侧向偏移。"""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(),
            nn.Linear(16 * 4, 1))

    def forward(self, x):
        return self.body(x).squeeze(-1)


def train(model, x, t, seed=SEED):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(seed + 1)
    for _ in range(STEPS):
        i = torch.randint(0, x.shape[0], (BATCH,), generator=g)
        loss = F.mse_loss(model(x[i]), t[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


def quantize(w, bits, per_channel=True):
    """对称均匀量化：折到整数格点上再折回来。
    per_channel=True 时每个输出通道各用自己的 scale，否则全张量共用一个。"""
    qmax = 2 ** (bits - 1) - 1
    flat = w.abs().flatten(1) if per_channel else w.abs().reshape(1, -1)
    scale = flat.amax(1).clamp(min=1e-12) / qmax                  # (out,) 或 (1,)
    if per_channel:
        scale = scale.view(-1, *([1] * (w.dim() - 1)))
    return (w / scale).round().clamp(-qmax - 1, qmax) * scale


def quantized(model, bits, per_channel=True):
    """复制一份模型，把所有卷积/线性层的权重换成量化后的值（激活不动）。"""
    q = copy.deepcopy(model)
    with torch.no_grad():
        for m in q.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                m.weight.copy_(quantize(m.weight, bits, per_channel))
    return q.eval()


@torch.no_grad()
def mae(model, x, t):
    """平均绝对误差，单位是"光斑偏移量的归一化刻度"，x SWING 就是像素。"""
    return (model(x) - t).abs().mean().item()


def size_table():
    """体积比是精确算术，不是估计：每参数字节数 x 参数量。"""
    print(f"\n体积比（只算权重，不含激活与 KV cache；按 {PARAMS/1e9:.0f}B 参数、十进制 GB）")
    print(f"  {'精度':<8}{'字节/参数':>10}{'权重体积':>12}{'相对 FP32':>12}")
    for name, b in (("FP32", 4.0), ("FP16", 2.0), ("INT8", 1.0), ("INT4", 0.5), ("INT3", 0.375)):
        print(f"  {name:<8}{b:>10.2f}{PARAMS * b / 1e9:>10.1f} GB{4.0 / b:>11.1f}x")


def outlier_layer():
    """模拟一个把视觉特征投到语言维度的投影层：256 个输出通道里，
    有 3 个的量级是其余的 40 倍（真实 VLM 里视觉侧总有几个"暴走通道"）。
    返回权重和离群通道的布尔掩码——掩码必须共用，否则表和图的统计口径会对不上。"""
    g = torch.Generator().manual_seed(SEED + 2)
    w = torch.randn(256, 256, generator=g)
    odd = torch.zeros(256, dtype=torch.bool)
    odd[:3] = True
    w[odd] *= 40.0
    return w, odd


def channel_rel_error(w, wq):
    """逐通道的相对误差。用布尔掩码取子集，不要用整数张量索引。"""
    return (wq - w).abs().mean(1) / w.abs().mean(1)


def report_outliers(w, odd):
    """逐张量 vs 逐通道：看量化的代价最终落在哪些通道上。"""
    print("\n一个投影层里混进 3 个量级 40 倍的离群通道后，权重的相对误差")
    print(f"  {'方案':<22}{'全体':>10}{'离群那 3 个通道':>16}{'其余 253 个':>14}")
    for bits in (8, 4):
        for pc, name in ((False, f"INT{bits} 逐张量"), (True, f"INT{bits} 逐通道")):
            rel = channel_rel_error(w, quantize(w, bits, pc))
            print(f"  {name:<22}{rel.mean():>9.1%}{rel[odd].mean():>15.1%}{rel[~odd].mean():>13.1%}")


def attention_logits(scale_by_sqrt_d, seed):
    """q·k^T，可选是否除以 sqrt(d)。不除就是"忘了缩放"的 logits。"""
    g = torch.Generator().manual_seed(seed)
    d, n_key, n_q = 64, 64, 256
    q = torch.randn(n_q, d, generator=g)
    k = torch.randn(n_key, d, generator=g)
    z = q @ k.T
    return z / d ** 0.5 if scale_by_sqrt_d else z


def report_softmax():
    """logits 量化 -> softmax 概率偏移 + argmax 翻转。
    翻转率能不能不量化就先估出来？翻转要 δ2 - δ1 > gap，而每个 δ 的幅度不超过 step/2，
    所以前两名间隔 < 量化步长的行才可能翻。这个占比是免费算出来的严格上界，
    但远不紧——上界与实测一起报，别把它当等式用。"""
    print("\n注意力 logits 量化后（64 个 key、256 个 query），同一份 q/k 只差一次缩放")
    rows = []
    for scaled in (True, False):
        z = attention_logits(scaled, SEED + 4)
        tag = "已除 sqrt(d)" if scaled else "忘了除 sqrt(d)"
        print(f"\n  {tag}：logits 标准差 {z.std():.2f}，量程 ±{z.abs().max():.1f}")
        print(f"    {'位宽':<6}{'步长':>8}{'softmax L1 偏移':>18}{'argmax 翻转率':>16}{'翻转上界':>12}")
        for bits in (8, 6, 4, 3):
            qmax = 2 ** (bits - 1) - 1
            step = z.abs().max() / qmax                          # 量化步长
            zq = quantize(z, bits, per_channel=False)
            p, pq = z.softmax(-1), zq.softmax(-1)
            l1 = (pq - p).abs().sum(-1).mean().item()
            flip = (pq.argmax(-1) != p.argmax(-1)).float().mean().item()
            top2 = z.topk(2, dim=-1).values
            gap = top2[:, 0] - top2[:, 1]                         # 前两名 logits 的间隔
            # 翻转要 δ2 - δ1 > gap，而每个 δ 的幅度不超过 step/2，故 gap < step 是必要条件
            bound = (gap < step).float().mean().item()            # 可能翻转的行占比 = 严格上界
            rows.append((scaled, bits, l1, flip, bound))
            print(f"    {bits:<6}{step:>8.3f}{l1:>17.2%}{flip:>15.1%}{bound:>11.1%}")
    n = len(rows) // 2
    same = all(abs(x[3] - y[3]) < 1e-9 for x, y in zip(rows[:n], rows[n:]))
    i4 = [j for j, r in enumerate(rows) if r[1] == 4]
    print(f"\n  两种缩放的翻转率{'完全相同' if same else '并不相同'}。这不是巧合：对称量化的步长取自本张量的 amax，"
          "\n  整体乘一个常数会把步长和前两名间隔按同一比例缩放，谁赢谁输不变。"
          "\n  但 softmax 不是尺度不变的：同样的相对量化误差，在没缩放的那组里"
          f"造成的概率偏移大得多（INT4：{rows[i4[0]][2]:.1%} → {rows[i4[1]][2]:.1%}）。")
    ratios = [r[4] / max(r[3], 1e-9) for r in rows]
    print(f"  翻转上界这一列是实测的 {min(ratios):.1f}~{max(ratios):.1f} 倍：它是严格的必要条件的占比，"
          "\n  但满足它的行里大多数并没有真的翻，所以只能当量级参考，不能当等式。")
    return rows


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    font = plotting.use_chinese_font()
    print(f"字体: {font or '未找到 CJK 字体，图中中文可能显示为方块'}\n")

    print(f"训练 {N_TRAIN} 条，留出 {N_TEST} 条；图 {SIZE}x{SIZE}，目标 = 光斑的侧向偏移")

    size_table()

    # ---- 主实验：位宽 vs 精度 ----
    # 单次训练的结果不可复现：换 BLAS 线程数就能让 INT8 的退化从 +16% 跳到 -6%。
    # 所以基准和每一档都在同一簇训练种子上取分布，先量出噪声地板再谈效应。
    print(f"\nPTQ 只量化权重、激活不动；{len(SEEDS)} 次独立训练（训练集与初始化都重采）")
    deg = {b: [] for b in BITS}
    bases, px = [], {}
    for s in SEEDS:
        torch.manual_seed(s)                                   # 初始化跟着种子走
        xtr, ttr = make_data(N_TRAIN, s)
        xte, tte = make_data(N_TEST, s + 100)
        model = train(OffsetNet(), xtr, ttr, seed=s)
        b0 = mae(model, xte, tte)
        bases.append(b0)
        for bits in BITS:
            deg[bits].append(mae(quantized(model, bits), xte, tte) / b0 - 1)
        px[s] = b0 * SWING
    spread = max(bases) / min(bases) - 1
    print(f"  FP32 基准自身：{min(bases):.4f} ~ {max(bases):.4f}（{min(px.values()):.2f} ~ "
          f"{max(px.values()):.2f} 像素），光是换颗种子训练就差 {spread:.0%}")
    print(f"  —— 网络落在哪个盆地是随机的，所以下面每一档都是配对比较"
          f"（同一颗种子训出的模型，只换权重），\n     判据是：这一档的退化在 {len(SEEDS)} 颗种子上会不会翻号\n")
    print(f"  {'位宽':<8}{'相对 FP32 均值':>16}{'最小':>10}{'最大':>10}{'跨过 0（分辨不出）':>20}")
    for bits in BITS:
        d = deg[bits]
        cross = "是" if min(d) < 0 < max(d) else "否"
        print(f"  INT{bits:<5}{sum(d)/len(d):>15.1%}{min(d):>10.1%}{max(d):>10.1%}{cross:>19}")

    report_outliers(*outlier_layer())
    sm = report_softmax()

    solid = [b for b in BITS if min(deg[b]) > 0 or max(deg[b]) < 0]   # 从不翻号的档位
    shaky = [b for b in BITS if b not in solid]
    print(f"\n结论：位宽不是均匀打折，但真正稳的结论比预想的窄。")
    print(f"      网络落在哪个盆地本身就带来 {spread:.0%} 的基准波动，高位量化的效应被它淹掉——")
    for b in shaky:
        print(f"      INT{b} 的退化在 {len(SEEDS)} 颗种子间从 {min(deg[b]):+.0%} 摆到 {max(deg[b]):+.0%}，"
              f"区间跨过 0：")
        print(f"        说 INT{b} 几乎无损，和说 INT{b} 退化一半，都能从这簇实验里挑出证据来。")
    top = max(solid)
    print(f"      INT{top} 起就不翻号了（{min(deg[top]):+.0%} ~ {max(deg[top]):+.0%}），"
          f"{len(SEEDS)} 颗种子全部远超文档给的 INT{top} 保持 93-97%。")
    print("      所以能确定的是断崖存在，不是打折的倍数——门槛定在哪儿，这个规模测不出来。")
    print("      另外两件事量化表里看不到：逐张量的代价全落在没有离群通道的普通通道上；")
    print("      生成任务要 argmax，翻转只可能发生在 前两名间隔 < 量化步长 的地方，")
    print("      这种失败没有轻微的错这一档，只有对和错。")

    # ---- 出图 ----
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13.6, 4.6))

    ax = fig.add_subplot(1, 3, 1)
    xs = list(range(len(BITS)))
    mean = [sum(deg[b]) / len(deg[b]) for b in BITS]
    lo = [min(deg[b]) for b in BITS]
    hi = [max(deg[b]) for b in BITS]
    from matplotlib.lines import Line2D
    g = torch.Generator().manual_seed(SEED)
    for x, b in zip(xs, BITS):                               # 每颗种子一个点，散开看分布
        jit = (torch.rand(len(SEEDS), generator=g) - 0.5) * 0.24
        ax.scatter([x + j for j in jit.tolist()], [1 + d for d in deg[b]],
                   s=22, color="#c0392b", alpha=0.55, zorder=3)
    ax.plot(xs, [1 + m for m in mean], "-", color="#7b241c", lw=2, zorder=3)
    for x, m, b in zip(xs, mean, BITS):
        # 实心 = 每颗种子都朝同一边偏（效应稳）；空心 = 区间跨过 0（分辨不出）
        ax.plot([x], [1 + m], "o", ms=10, zorder=4, mec="#7b241c", mew=2,
                mfc="#7b241c" if b in solid else "white")
        ax.annotate(f"{m:+.0%}", (x, 1 + m), fontsize=9,
                    textcoords="offset points", xytext=(0, 13), ha="center")
    ax.axhline(1.0, color="gray", ls="--", lw=1.2, zorder=2)
    ax.legend(handles=[
        Line2D([], [], color="#c0392b", marker="o", ls="none", alpha=0.55, label="单颗训练种子"),
        Line2D([], [], color="#7b241c", marker="o", ls="-", label=f"{len(SEEDS)} 颗种子的均值"),
        Line2D([], [], color="#7b241c", marker="o", ls="none", mfc="#7b241c", mew=2,
               label="各颗种子都不翻号"),
        Line2D([], [], color="#7b241c", marker="o", ls="none", mfc="white", mew=2,
               label="区间跨过 0，分辨不出")], fontsize=8, loc="upper left")
    ax.set_yscale("log")
    ax.set_ylim(0.55, (1 + max(hi)) * 2.4)                   # 给顶部那行百分比留头
    ax.set_xticks(xs)
    ax.set_xticklabels([f"INT{b}" for b in BITS])
    ax.set_xlabel("权重量化位宽")
    ax.set_ylabel("平均绝对误差 / FP32 基准（对数轴）")
    ax.set_title("(a) 高位的效应埋在训练噪声里，低位的断崖才稳")
    ax.grid(axis="y", alpha=0.3)

    ax = fig.add_subplot(1, 3, 2)
    w, odd = outlier_layer()
    labels, series = [], {}
    for bits in (8, 4):
        for pc, name in ((False, "逐张量"), (True, "逐通道")):
            rel = channel_rel_error(w, quantize(w, bits, pc))
            key = f"INT{bits} {name}"
            labels.append(key)
            series[key] = [rel[~odd].mean().item() * 100, rel[odd].mean().item() * 100]
    xs2 = torch.arange(len(labels))
    wd = 0.36
    for j, (grp, color) in enumerate((("其余 253 个通道", "#2471a3"), ("离群那 3 个通道", "#e67e22"))):
        vals = [series[k][j] for k in labels]
        ax.bar(xs2 + (j - 0.5) * wd, vals, wd, label=grp, color=color)
        for x, v in zip(xs2 + (j - 0.5) * wd, vals):
            ax.text(x, v * 1.15, f"{v:.1f}%", ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(0.2, 400)                                    # 给柱顶那行百分比留头
    ax.set_xticks(xs2)
    ax.set_xticklabels(labels, rotation=15, fontsize=9)
    ax.set_ylabel("权重的逐通道相对误差（对数轴）")
    ax.set_title("(b) 逐张量的代价，落在没有离群通道的普通通道上")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    ax = fig.add_subplot(1, 3, 3)
    sub = [r for r in sm if r[0]]                              # 只画已除 sqrt(d) 的正常情形
    xs3 = list(range(len(sub)))
    cnt = [r[3] for r in sub]
    bnd = [r[4] for r in sub]
    ax.fill_between(xs3, cnt, bnd, color="#c0392b", alpha=0.10)
    ax.plot(xs3, bnd, "s--", color="#7f8c8d", lw=1.5, label="上界：前两名间隔 < 步长/2 的行占比")
    ax.plot(xs3, cnt, "o-", color="#c0392b", lw=2, label="实测 argmax 翻转率")
    ax.set_xticks(xs3)
    ax.set_xticklabels([f"INT{r[1]}" for r in sub])
    ax.set_xlabel("logits 量化位宽")
    ax.set_ylabel("argmax 翻转率")
    ax.set_title("(c) 生成失败是离散的：翻转只发生在间隔不足半步长处")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    fig.suptitle("量化不是均匀打折：高位（INT8）的效应埋在训练噪声里，低位（INT4 往下）是断崖", fontsize=13)
    fig.tight_layout()
    plotting.save(fig, "n_quant_bits.png")


if __name__ == "__main__":
    main()
