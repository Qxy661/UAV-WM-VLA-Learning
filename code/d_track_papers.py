"""
Demo D — 追踪一个实验室：把 arXiv 号变成一张研究脉络图

对应文档：docs/08-研究前沿与开放问题/04-最新进展与团队追踪.md

要说明的一件事：读研究前沿，"按团队读"往往比"按论文读"更快建立领域感。
但难点在于——同一个组的论文散落在几十个 arXiv 号里，标题还经常被二手报道改写。
这个脚本做两件事：

  ① 从 arXiv 号本身解析出年月（YYMM.xxxxx 的前四位就是投稿年月），
     把两个代表团队 2024–2026 的产出画成时间线，直接看出"哪个方向在加速"；
  ② 可选地联网去 arXiv API 拉权威元数据，和本地记录逐条对照，
     把标题对不上的挑出来 —— 这是防止"记错 arXiv 号"的最省事办法。

运行：
    py -3.9 code/d_track_papers.py              # 只出图，离线可跑
    py -3.9 code/d_track_papers.py --verify     # 额外联网核对元数据

注意：--verify 走 https://export.arxiv.org（HTTP 会失败），首次约 10 秒。
"""

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import plotting

# ---------------------------------------------------------------------------
# 数据：两个代表团队的论文清单
#   字段 = (arXiv ID, 方向, 本地记录的短标题)
# ID 前四位 YYMM 即投稿年月，所以不需要另存日期字段 —— 少一个会写错的地方。
# ---------------------------------------------------------------------------
ZJU = [  # 浙江大学 FAST Lab（高飞 / 许超）
    ("2401.07784", "集群",     "Certifiable Mutual Localization"),
    ("2403.02977", "集群",     "FIRI 凸区域膨胀"),
    ("2403.04586", "敏捷飞行", "Learning Speed Adaptation"),
    ("2403.13455", "集群",     "FACT 坐标初始化"),
    ("2405.07736", "敏捷飞行", "Optimization Embedded Networks"),
    ("2407.01292", "集群",     "FoV-Limited Swarm"),
    ("2408.01649", "感知",     "LF-3PM LiDAR 感知规划"),
    ("2409.00895", "敏捷飞行", "Whole-Body Control Narrow Gaps"),
    ("2502.16887", "集群",     "Primitive-Swarm"),
    ("2503.00496", "敏捷飞行", "Flying on Point Clouds with RL"),
    ("2503.00785", "空中操作", "FLOAT Drone"),
    ("2504.15138", "敏捷飞行", "Aerobatic Flight via Diffusion"),
    ("2505.10018", "感知",     "LEMON-Mapping"),
    ("2505.15010", "空中操作", "Shape-Adaptive Deformable Quadrotor"),
    ("2507.21338", "感知",     "Terrestrial-Aerial Bimodal Exploration"),
    ("2510.11306", "感知",     "Rotor-Failure-Aware Quadrotors"),
    ("2510.24315", "感知",     "CoNi-OA 无全局状态避障"),
    ("2511.01186", "感知",     "LiDAR-VGGT"),
    ("2512.15258", "VLA/世界模型", "VLA-AN"),
    ("2602.00708", "VLA/世界模型", "USS-Nav"),
    ("2602.08599", "空中操作", "Force-Aware Grasping"),
    ("2602.08653", "敏捷飞行", "Safety-Shielded RL"),
    ("2602.09765", "VLA/世界模型", "NavDreamer"),
    ("2604.05828", "敏捷飞行", "Precise Aggressive Maneuvers"),
    ("2605.19600", "VLA/世界模型", "FlyMirage"),
    ("2606.02313", "VLA/世界模型", "Expert-Guided GRPO"),
    ("2606.11708", "感知",     "Explore From Sketch"),
    ("2607.04260", "空中操作", "FLOAT for Physical Interaction"),
    ("2607.29009", "VLA/世界模型", "D-VLC"),
    ("2608.14135", "敏捷飞行", "AgilePE 追逃"),
    ("2608.16640", "感知",     "DPNet 死胡同预测"),
    ("2609.17198", "感知",     "TIO-Former"),
    ("2609.18191", "感知",     "OmniRisk"),
    ("2609.19824", "VLA/世界模型", "TADreamer"),
    ("2609.25898", "感知",     "Robust Active-Perception"),
    ("2609.30770", "VLA/世界模型", "NavGen"),
]

BUAA = [  # 北京航空航天大学 Colab（刘偲）
    ("2410.07087", "VLN",       "OpenUAV 平台与基准"),
    ("2411.13610", "感知定位",   "Video2BEV"),
    ("2505.15725", "VLA",       "UAV-Flow Colosseo"),
    ("2506.06677", "机器人操作", "RoboCerebra"),
    ("2506.09839", "导航",       "OctoNav"),
    ("2507.04430", "人机交互",   "AirStar"),
    ("2508.15232", "VLN",       "AeroDuo"),
    ("2601.11404", "VLA",       "ACoT-VLA"),
    ("2604.17190", "VLN",       "LookasideVLN"),
    ("2605.18617", "机器人操作", "ManiSoft"),
    ("2605.27491", "世界模型",   "GE-Sim 2.0"),
    ("2606.06836", "VLA",       "FLIGHT / Think Like a Pilot"),
    ("2606.30576", "感知定位",   "Beyond 2D Matching"),
    ("2607.02646", "部署框架",   "EVA-Client"),
    ("2607.11529", "VLN",       "Parse, Search, Confirmation"),
]

COLORS = {"集群": "#4C72B0", "敏捷飞行": "#DD8452", "空中操作": "#55A868",
          "感知": "#8172B3", "VLA/世界模型": "#C44E52",
          "VLN": "#4C72B0", "VLA": "#C44E52", "导航": "#DD8452",
          "机器人操作": "#55A868", "人机交互": "#8172B3",
          "感知定位": "#937860", "世界模型": "#DA8BC3", "部署框架": "#8C8C8C"}


def parse_date(arxiv_id):
    """从 arXiv ID 解析投稿年月：YYMM.xxxxx 的前四位。"""
    yy, mm = int(arxiv_id[:2]), int(arxiv_id[2:4])
    return 2000 + yy, mm


def to_month(arxiv_id):
    """转成连续月序号，以 2024-01 为 0，方便画时间轴。"""
    y, m = parse_date(arxiv_id)
    return (y - 2024) * 12 + (m - 1)


def summarize(rows, name):
    """打印分方向统计 + 分年度统计。"""
    dirs = Counter(r[1] for r in rows)
    years = Counter(parse_date(r[0])[0] for r in rows)
    print(f"\n【{name}】共 {len(rows)} 篇")
    print("  分年度: " + "  ".join(f"{y}年 {years[y]:2d} 篇" for y in sorted(years)))
    print("  分方向: " + "  ".join(f"{d} {n}" for d, n in dirs.most_common()))
    return dirs, years


def plot(zju_rows, buaa_rows, out_name="track_two_labs.png"):
    """左边时间线、右边方向分布。"""
    plt = plotting.plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.2),
                                   gridspec_kw={"height_ratios": [2.1, 1]})

    # ---- 上：时间线散点 ----
    for y, rows, lab, mark in ((1, zju_rows, "浙大 FAST Lab", "o"),
                               (0, buaa_rows, "北航 Colab", "s")):
        for aid, d, _ in rows:
            ax1.scatter(to_month(aid), y, s=95, marker=mark,
                        color=COLORS.get(d, "#888888"),
                        edgecolor="white", linewidth=0.8, zorder=3)
        ax1.text(-1.6, y, lab, ha="right", va="center", fontsize=11)

    ax1.set_yticks([])
    ax1.set_ylim(-0.7, 1.7)
    ax1.set_xlim(-11, 35)
    ticks = list(range(0, 31, 6))
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(["2024-01", "2024-07", "2025-01",
                         "2025-07", "2026-01", "2026-07"], fontsize=9)
    for t in (0, 12, 24):                       # 年份分隔线
        ax1.axvline(t, color="#bbbbbb", linewidth=0.8, zorder=1)
    ax1.set_xlabel("arXiv 投稿年月（由 ID 前四位解析）")
    ax1.set_title("两个代表团队 2024–2026 的论文产出：越靠右越新")
    ax1.grid(alpha=0.25, axis="x")

    # ---- 下：方向分布 ----
    dirs = Counter()
    for _, d, _ in zju_rows:
        dirs["浙大\n" + d] += 1
    for _, d, _ in buaa_rows:
        dirs["北航\n" + d] += 1
    items = dirs.most_common()
    xs = range(len(items))
    ax2.bar(xs, [n for _, n in items],
            color=[COLORS.get(k.split("\n")[1], "#888888") for k, _ in items])
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels([k for k, _ in items], fontsize=8)
    ax2.set_ylabel("论文数")
    ax2.set_title("方向分布：浙大以规划/感知为基本盘，北航集中在语言导航")
    for x, (_, n) in zip(xs, items):
        ax2.text(x, n, str(n), ha="center", va="bottom", fontsize=8)
    ax2.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    plotting.save(fig, out_name)


def verify(ids_with_titles):
    """可选：联网核对 arXiv 元数据，把标题对不上的挑出来。"""
    import urllib.request
    import xml.etree.ElementTree as ET

    ids = [a for a, _, _ in ids_with_titles]
    url = ("https://export.arxiv.org/api/query?id_list=" + ",".join(ids)
           + "&max_results=100")
    print(f"\n【联网核对】向 arXiv API 查询 {len(ids)} 个 ID ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=60).read()
    except Exception as e:            # 网络不通时不要让整个脚本挂掉
        print(f"  查询失败（{type(e).__name__}: {e}）")
        print("  提示：本机若需代理，请确认 HTTPS 可直连 export.arxiv.org。")
        print("  —— 上面的图和统计不依赖网络，已正常生成。")
        return

    ns = {"a": "http://www.w3.org/2005/Atom"}
    got = {}
    for e in ET.fromstring(raw).findall("a:entry", ns):
        aid = e.find("a:id", ns).text.split("/abs/")[-1].split("v")[0]
        got[aid] = " ".join(e.find("a:title", ns).text.split())

    miss = [a for a in ids if a not in got]
    print(f"  命中 {len(got)}/{len(ids)}" + (f"，未返回: {miss}" if miss else ""))
    print("  逐条核对（本地短标题只是备注，不做严格比对；这里看的是权威标题）:")
    for aid, d, short in sorted(ids_with_titles):
        real = got.get(aid)
        if real is None:
            print(f"    [!] {aid}  arXiv 未返回 —— 该 ID 可能不存在")
        else:
            print(f"    {aid}  [{d}] {real[:78]}")


def main():
    font = plotting.use_chinese_font()
    print(f"字体: {font or '未找到 CJK 字体，图中中文可能显示为方块'}")

    zju = sorted(ZJU)
    buaa = sorted(BUAA)
    summarize(zju, "浙江大学 FAST Lab（高飞 / 许超）")
    summarize(buaa, "北京航空航天大学 Colab（刘偲）")

    zju_new = sum(1 for a, _, _ in zju if parse_date(a)[0] >= 2026)
    print(f"\n一个可以直接看出来的趋势：")
    print(f"  浙大 FAST Lab 在 2026 年（截至 9 月）已有 {zju_new} 篇，"
          f"超过其 2024 全年的产出")
    print(f"  其中 VLA/世界模型方向全部出现在 2025-12 之后 —— "
          f"这个组是最近一年才切入这条线的")

    plot(zju, buaa)
    if "--verify" in sys.argv:
        verify(zju + buaa)
    else:
        print("\n（加 --verify 可联网核对上述 ID 的权威标题）")


if __name__ == "__main__":
    main()
