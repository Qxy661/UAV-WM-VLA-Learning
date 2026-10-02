"""
Demo O — 显存这笔账，为什么同一件事各处算出来不一样

对应文档：docs/07-实践指南/01-环境搭建.md、02-复现指南-UAV-Flow.md、03-复现指南-CognitiveDrone.md、
          04-复现指南-FlightDiffusion.md、06-复现指南-GeoChat.md、07-复现指南-DreamerV3-Drone.md、
          08-可复现项目候选清单.md，以及 docs/03-VLA专题/05-机载部署与优化.md、
          docs/04-VLM专题/04-边缘VLM部署.md

要验证的结论：同一个 7B 模型的 LoRA 微调，一处写 18 GB、一处 24 GB、一处 40 GB。
这些数不是谁抄错了，是四项开销（权重 / 梯度 / 优化器状态 / 激活）算了几项、各按什么
精度算。更关键的是那张表里少了一个自变量：每个样本多少个 token——九张表一张都没写它。

做法：
  1. 权重 / 梯度 / 优化器状态是精确算术，直接给公式
  2. 优化器状态真建一个模型、真走一步 Adam，按 state 里的字节数实测
  3. 激活用 saved_tensors_hooks 拦下反向图保留的每个张量。有个坑：参数本身也会被拦下，
     必须按 storage 指针剔除，否则固定项会重复计入
  4. 拿文档里只差 batch 的两行做减法，反解出「固定项」与「每样本项」两个未知数，
     不需要任何额外信息——这个算式可以用来审任何一张显存表

运行：py -3.9 code/o_budget.py    # 约 2 分钟，CPU 即可，只需 torch 与 torchvision
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import plotting

SEED = 0
GB = 2 ** 30                  # 按 2 的幂算，与 nvidia-smi 显示的 MiB 口径一致

# 7B 级 VLM 的典型结构（LLaMA-2 7B）：32 层、隐藏 4096、32 个头
N_LAYER, D_MODEL, N_HEAD = 32, 4096, 32
N_MATRIX_PER_LAYER = 7        # q/k/v/o/gate/up/down，GeoChat 6.2 的 target_modules

P_7B = 7.0e9
P_RESNET50 = 25.6e6

# 激活实测用的规模。SEQ 固定成 512，好和 7B 那张表按 token 数换算
SEQ = 512
GRID_LAYERS = (1, 2, 4)
GRID_BATCH = (1, 2, 4)
SEQ_SWEEP = (512, 1024, 2048)
RESNET_BATCH = (16, 32)


def lora_params(r):
    """LoRA 参数量：每层 7 个 d x d 投影，各挂一对 d x r 的低秩矩阵。"""
    return N_LAYER * N_MATRIX_PER_LAYER * 2 * D_MODEL * r


def display_width(s):
    """字符串在等宽终端里占几格。中日韩字符占两格，len() 数不出来。"""
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def pad(s, width, right=False):
    """按显示宽度补空格。str.ljust 数的是字符数，中文列直接用它会对不齐。"""
    blank = " " * max(0, width - display_width(s))
    return blank + s if right else s + blank


def static_gb(n_frozen, n_trainable):
    """权重 / 梯度 / 优化器状态：三份都是精确算术，没有任何假设。
    冻结的权重只存 fp16 一份；可训练的参数要存 fp16 权重 + fp16 梯度 + fp32 两份动量。"""
    return (2 * n_frozen + 12 * n_trainable) / GB


def static_table():
    print("一、静态三项（权重 / 梯度 / 优化器状态）是精确算术，算不出花来")
    print("  " + pad("配置", 26) + pad("权重", 8, True) + pad("梯度", 8, True)
          + pad("Adam", 8, True) + pad("合计", 8, True) + pad("字节/参数", 10, True))
    rows = [
        ("全量 fp16 + Adam", 0.0, P_7B, 2.0),
        ("全量 fp32 + Adam", 0.0, P_7B, 4.0),
        ("只推理, fp16", P_7B, 0.0, 2.0),
        (f"LoRA r=16 ({lora_params(16)/1e6:.0f}M 可训练)", P_7B, lora_params(16), 2.0),
        (f"LoRA r=64 ({lora_params(64)/1e6:.0f}M 可训练)", P_7B, lora_params(64), 2.0),
    ]
    for name, nf, nt, bpp in rows:
        w, g, a = bpp * (nf + nt) / GB, bpp * nt / GB, 8 * nt / GB
        print(f"  {pad(name, 26)}{w:>7.1f}G{g:>7.1f}G{a:>7.1f}G{w+g+a:>7.1f}G{(w+g+a)*GB/P_7B:>10.0f}")
    print("  注：混合精度还要留一份 fp32 主权重（+4 字节/参数），上表未计。")
    print("      只推理那一行只有权重一列——同一句话在训练和推理下能差 8 倍，就是这么来的。")


def measure_adam():
    """真走一步 Adam，数 state 里存了多少字节。不推公式，直接数。"""
    torch.manual_seed(SEED)
    m = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    m(torch.randn(16, 256)).sum().backward()
    opt.step()
    n = sum(p.numel() for p in m.parameters())
    b = sum(v.numel() * v.element_size()
            for st in opt.state.values() for v in st.values() if torch.is_tensor(v))
    print(f"\n二、优化器状态不是估的：真建一个 {n/1e3:.0f}K 参数的模型，真走一步 Adam")
    print(f"  实测 {b} 字节 / {n} 参数 = {b/n:.1f} 字节每可训练参数"
          f"（一阶动量 fp32 + 二阶动量 fp32，正好 8）")
    print(f"  换算：7B 全量微调这一项要 {8*P_7B/GB:.1f} GB；")
    print(f"        换成 LoRA r=16，可训练参数只剩 {lora_params(16)/P_7B:.2%}，"
          f"这一项塌到 {8*lora_params(16)/GB:.2f} GB。")
    return b / n


def activation_bytes(module, x):
    """拦下 autograd 为反向传播保留的每个张量。

    坑在这里：Linear 的反向要用到权重，所以参数张量也会被 pack 收到——但它们本来就在
    显存里，不是激活新增的。按 storage 指针把参数剔除，剩下的才是激活真正的增量。
    不剔的话，一个 4096 宽的块会凭空多出 768 MB 的假激活。"""
    params = {p.untyped_storage().data_ptr() for p in module.parameters()}
    total = 0

    def pack(t):
        nonlocal total
        if t.untyped_storage().data_ptr() not in params:
            total += t.numel() * t.element_size()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        module(x).sum()
    return total


class Block(nn.Module):
    """一层标准 Transformer：注意力 + MLP，两处残差。"""

    def __init__(self, d, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class FusedAttn(nn.Module):
    """包一层只是为了让签名和 ManualAttn 一致，好放进同一个测量函数。"""

    def __init__(self, d, heads):
        super().__init__()
        self.a = nn.MultiheadAttention(d, heads, batch_first=True)

    def forward(self, x):
        return self.a(x, x, x, need_weights=False)[0]


class ManualAttn(nn.Module):
    """同样一层，但把 softmax(QK^T)V 手写出来——注意力矩阵留在图里，退不掉。"""

    def __init__(self, d, heads):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.h, self.dh = heads, d // heads

    def forward(self, x):
        b, n, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.view(b, n, self.h, self.dh).transpose(1, 2)
        k = k.view(b, n, self.h, self.dh).transpose(1, 2)
        v = v.view(b, n, self.h, self.dh).transpose(1, 2)
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.dh), -1)
        return self.proj((a @ v).transpose(1, 2).reshape(b, n, -1))


def measure_resnet():
    """ResNet-50 直接对上 07/03 那张表，而且是 torchvision 的真实结构。"""
    model = torchvision.models.resnet50(weights=None)
    st = static_gb(0, P_RESNET50)
    print(f"\n三、激活：ResNet-50（{P_RESNET50/1e6:.1f}M 参数，torchvision 原版结构）")
    print("  " + pad("batch", 8) + pad("权重+梯度+Adam", 15, True)
          + pad("激活", 9, True) + pad("合计", 9, True))
    got = {}
    for bs in RESNET_BATCH:
        x = torch.randn(bs, 3, 224, 224, requires_grad=True)
        a = activation_bytes(model, x)
        got[bs] = a
        print(f"  {pad(str(bs), 8)}{st:>14.2f}G{a/GB:>8.2f}G{st + a/GB:>8.2f}G")
    per = (got[RESNET_BATCH[-1]] - got[RESNET_BATCH[0]]) / (RESNET_BATCH[-1] - RESNET_BATCH[0])
    fixed = got[RESNET_BATCH[0]] - per * RESNET_BATCH[0]
    print(f"  两行相减解出：固定项 {fixed/GB:+.2f} GB，每样本 {per/GB:.3f} GB。")
    print(f"  固定项是 0 而不是参数那 {4*P_RESNET50/GB:.2f} GB——因为参数已经在上一段被剔掉了，")
    print("  激活里本来就不该有它。两笔账各归各的，才不会重复计入。")
    return fixed, per


def measure_grid():
    """激活随层数与 batch 的增长，验证它在两个方向上都严格线性。"""
    print(f"\n四、激活随规模怎么涨（隐藏 {D_MODEL}、{SEQ} 个 token）")
    print("  " + pad("层数", 6) + pad("batch", 7, True) + pad("激活", 11, True)
          + pad("每层每样本", 13, True))
    res = {}
    for nl in GRID_LAYERS:
        torch.manual_seed(SEED)
        net = nn.Sequential(*[Block(D_MODEL, N_HEAD) for _ in range(nl)])
        res[nl] = {}
        for bs in GRID_BATCH:
            x = torch.randn(bs, SEQ, D_MODEL, requires_grad=True)
            t = activation_bytes(net, x)
            res[nl][bs] = t
            print(f"  {pad(str(nl), 6)}{bs:>7}{t/2**20:>9.1f}MB{t/(nl*bs)/2**20:>11.1f}MB")
    per_layer_sample = res[1][1]
    vals = [res[nl][bs] / (nl * bs) for nl in GRID_LAYERS for bs in GRID_BATCH]
    print(f"  归一化之后九格里只差 {max(vals)/min(vals)-1:.1%}——层数和 batch 都是一次方，"
          f"所以能安全外推。")
    return per_layer_sample


def measure_seq():
    """序列长度那一维就不是一次方了：手写注意力会留下 seq^2 的矩阵。"""
    print("\n五、序列长度这一维：nn.MultiheadAttention 走 SDPA 融合核，手写版走朴素路径")
    print("  " + pad("序列长度", 10) + pad("融合核", 10, True) + pad("手写", 10, True)
          + pad("差值", 10, True) + pad("手写 KB/token", 14, True))
    fused = FusedAttn(D_MODEL, N_HEAD)
    manual = ManualAttn(D_MODEL, N_HEAD)
    got = {}
    for seq in SEQ_SWEEP:
        x = torch.randn(1, seq, D_MODEL)
        a = activation_bytes(fused, x.clone().requires_grad_(True))
        b = activation_bytes(manual, x.clone().requires_grad_(True))
        got[seq] = (a, b)
        print(f"  {pad(str(seq), 10)}{a/2**20:>8.1f}MB{b/2**20:>8.1f}MB"
              f"{(b-a)/2**20:>8.1f}MB{b/seq/2**10:>12.0f}KB")
    ss = SEQ_SWEEP
    ef = math.log(got[ss[-1]][0] / got[ss[0]][0]) / math.log(ss[-1] / ss[0])
    em = math.log(got[ss[-1]][1] / got[ss[0]][1]) / math.log(ss[-1] / ss[0])
    d1 = got[ss[1]][1] - got[ss[1]][0]
    d0 = got[ss[0]][1] - got[ss[0]][0]
    d2 = got[ss[2]][1] - got[ss[2]][0]
    print(f"  按幂律拟合：融合核是 {ef:.2f} 次方，手写版是 {em:.2f} 次方。")
    print(f"  再看差值那一列：{d0/2**20:.1f} → {d1/2**20:.1f} → {d2/2**20:.1f} MB，"
          f"每翻倍涨 {d1/d0:.2f} 倍和 {d2/d1:.2f} 倍。")
    print(f"  序列长度翻倍是 2 倍，差值涨 4 倍——这一项是 seq 的平方，"
          f"就是 FlashAttention 要消掉的东西。")
    return got


def reconcile(per_layer_sample, fixed, per_sample):
    """把文档的数减去静态账，余额除以实测的每 token 字节数，反解出它隐含的序列长度。"""
    per_token = per_layer_sample * N_LAYER / SEQ
    print("\n六、对账：把文档的数减去静态账，余额能反解出一个没写出来的自变量")
    print(f"  实测的 7B 激活：{per_token/2**20:.1f} MB/token（{N_LAYER} 层合计，与 batch 无关）")
    rows = [
        ("07/06 GeoChat", "LoRA r=64, bs=2", 18.0, P_7B, lora_params(64), 2),
        ("07/06 GeoChat", "LoRA r=128, bs=4", 24.0, P_7B, lora_params(128), 4),
        ("07/06 GeoChat", "全量微调, 4 卡", 80.0, 0.0, P_7B, 1),
        ("07/02 UAV-Flow", "LoRA, bs=4, fp16", 18.0, P_7B, lora_params(16), 4),
        ("07/02 UAV-Flow", "LoRA, bs=8, bf16", 40.0, P_7B, lora_params(16), 8),
    ]
    print("  " + pad("出处", 16) + pad("配置", 20) + pad("文档", 7, True)
          + pad("静态", 7, True) + pad("余额", 7, True) + pad("反解 token", 11, True))
    toks = []
    for src, cfg, doc, nf, nt, bs in rows:
        st = static_gb(nf, nt)
        tok = (doc - st) * GB / (per_token * bs)
        toks.append(tok)
        print(f"  {pad(src, 16)}{pad(cfg, 20)}{doc:>6.0f}G{st:>6.1f}G{doc-st:>6.1f}G{tok:>11.0f}")
    print(f"  反解出来的序列长度落在 {min(toks):.0f} ~ {max(toks):.0f} token，"
          f"跨度 {max(toks)/min(toks):.1f} 倍。")
    print("  这个区间正是 VLM 读一张图加一句指令的量级——没有一行是荒谬的，")
    print("  差的近 3 倍就藏在没人写的那个自变量里。")
    print("  有意思的是 GeoChat 全量微调那行：4 卡并没有让每卡省下什么，因为数据并行")
    print("  复制的是整个模型。它的 80 GB 几乎全是静态账 78.2 GB——这一行是对的。")

    print("\n  统一按 256 token 重算，GeoChat 那三行几乎重合，UAV-Flow 两行还差一截：")
    print("  " + pad("配置", 28) + pad("文档", 7, True) + pad("按 256 token 重算", 20, True)
          + pad("差", 9, True))
    for src, cfg, doc, nf, nt, bs in rows:
        conv = static_gb(nf, nt) + per_token * 256 * bs / GB
        print(f"  {pad(cfg, 28)}{doc:>6.0f}G{conv:>19.1f}G{conv/doc-1:>+9.0%}")

    print("\n  但 07/03 那张 ResNet-50 的表对不上，而且是另一种对不上：")
    st = static_gb(0, P_RESNET50)
    print("  " + pad("配置", 12) + pad("文档", 7, True) + pad("静态", 8, True)
          + pad("余额", 8, True) + pad("按实测应需", 12, True))
    for name, doc, bs in (("bs=16", 8.0, 16), ("bs=32", 14.0, 32)):
        need = fixed / GB + per_sample * bs / GB
        print(f"  {pad(name, 12)}{doc:>6.0f}G{st:>7.2f}G{doc-st:>7.2f}G{need:>11.2f}G")
    doc_per = (14.0 - 8.0) / (32 - 16)
    print(f"  这张表自己解出来是固定项 2.0 GB、每样本 {doc_per:.3f} GB，"
          f"而实测是固定项 0.00 GB、每样本 {per_sample/GB:.3f} GB。")
    print(f"  每样本差 {doc_per/(per_sample/GB):.1f} 倍，还凭空多出 2.0 GB 的固定项。")
    print(f"  小模型上参数只有 {st:.2f} GB，nvidia-smi 读到的东西里框架开销与显存碎片占了大头，")
    print("  那不是能按算法量算出来的。所以 7B 那几张表能和实测对上，这张对不上——两码事。")


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    font = plotting.use_chinese_font()
    print(f"字体: {font or '未找到 CJK 字体，图中中文可能显示为方块'}")

    static_table()
    measure_adam()
    fixed, per_sample = measure_resnet()
    per_layer_sample = measure_grid()
    seq_data = measure_seq()
    reconcile(per_layer_sample, fixed, per_sample)

    print("\n结论：显存表的每一项都是算得出来的，唯独「序列长度」这个自变量没人写。")
    print("      补上它，九张表里的数就能互相换算；补不上，它们就永远是九个孤立的经验值。")

    figure(fixed, per_sample, per_layer_sample, seq_data)


def figure(fixed, per_sample, per_layer_sample, seq_data):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    ax = axes[0]
    seqs = list(SEQ_SWEEP)
    f = [seq_data[s][0] / 2**20 for s in seqs]
    m = [seq_data[s][1] / 2**20 for s in seqs]
    ax.loglog(seqs, f, "o-", lw=2, color="#2166ac", label="融合核（SDPA）")
    ax.loglog(seqs, m, "s-", lw=2, color="#b2182b", label="手写 softmax(QK^T)V")
    ax.set_xticks(seqs)
    ax.set_xticklabels([str(s) for s in seqs])
    ax.minorticks_off()                                   # 不然 6x10^2 会和 512 叠在一起
    ax.set_xlabel("序列长度")
    ax.set_ylabel("保留到反向的激活（MB）")
    expo = math.log(m[-1] / m[0]) / math.log(seqs[-1] / seqs[0])
    ax.set_title(f"(a) 融合核是一次方，手写版是 {expo:.2f} 次方", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    tables = [
        ("DreamerV3\n64x64", 0.08, 10),
        ("FlightDiffusion\nhidden=256", 0.25, 24),
        ("CognitiveDrone\nResNet-50", 0.375, 38),
    ]
    for name, p, dy in tables:
        ax.scatter(p, 2.0, s=90, color="#b2182b", zorder=3)
        ax.annotate(name, (p, 2.0), textcoords="offset points",
                    xytext=(0, dy), ha="center", fontsize=8.5)
    ax.scatter(per_sample / GB, fixed / GB, s=180, marker="*", color="#2166ac", zorder=4)
    ax.annotate("(实测) ResNet-50", (per_sample / GB, fixed / GB),
                textcoords="offset points", xytext=(0, -20), ha="center", fontsize=8.5)
    ax.axhline(2.0, ls="--", lw=1, color="#888")
    ax.text(0.015, 1.80, "三张文档表解出来的固定项都是 2.0 GB", fontsize=8.5, color="#555")
    ax.set_xlabel("每样本激活（GB）")
    ax.set_ylabel("固定项（GB）")
    ax.set_xlim(-0.02, 0.46)
    ax.set_ylim(-0.3, 3.4)
    ax.set_title("(b) 两行相减解出的两个未知数：文档齐齐落在 2.0，实测落在 0", fontsize=11)
    ax.grid(alpha=0.3)

    ax = axes[2]
    per_token = per_layer_sample * N_LAYER / SEQ
    lo = static_gb(P_7B, lora_params(64))                 # static_gb 出来就是 GB，别再除一次
    configs = [
        ("7B 推理 fp16", 2 * P_7B / GB, 0.0),
        ("7B + LoRA\n256 token, bs=2", lo, per_token * 256 * 2 / GB),
        ("7B + LoRA\n512 token, bs=2", lo, per_token * 512 * 2 / GB),
        ("7B 全量 fp16+Adam", static_gb(0, P_7B), per_token * 256 / GB),
        ("ResNet-50 bs=32", static_gb(0, P_RESNET50), fixed / GB + per_sample * 32 / GB),
    ]
    names = [c[0] for c in configs]
    base = [c[1] for c in configs]
    act = [c[2] for c in configs]
    y = range(len(configs))
    ax.barh(y, base, color="#2166ac", label="权重/梯度/优化器")
    ax.barh(y, act, left=base, color="#f4a582", label="激活")
    for t in (24, 48, 80):
        ax.axvline(t, ls=":", color="#888", lw=1)
        ax.text(t, -0.62, f"{t}G 档", fontsize=8, color="#666", ha="center")
    for i, (b, a) in enumerate(zip(base, act)):
        ax.text(b + a + 1.5, i, f"{b+a:.0f}G", va="center", fontsize=8.5)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("显存（GB）")
    ax.set_xlim(0, 104)
    ax.set_ylim(len(configs) - 0.4, -1.0)                 # 上下留白给档位标签
    ax.set_title("(c) 静的是硬的，浮动全在激活那一截", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle("显存的四项开销：三项是精确算术，只有激活随规模走，而序列长度是那张表从没写过的自变量",
                 fontsize=13)
    fig.tight_layout()
    plotting.save(fig, "o_budget.png")


if __name__ == "__main__":
    main()
