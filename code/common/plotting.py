"""
出图工具：Agg 后端 + 中文字体，保证在无显示器的环境下也能存图。

Windows 上 matplotlib 默认字体不含汉字，图里会变方块。这里显式挑一个
系统中存在的 CJK 字体；一个都没有时退回英文标注而不是画一屏豆腐块。
"""

import matplotlib
matplotlib.use("Agg")           # 必须在 pyplot 之前设置

import matplotlib.pyplot as plt
from pathlib import Path

# code/common/plotting.py -> code/common -> code -> 仓库根
ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "figures"

# 常见 CJK 字体，按优先级排列
_CJK_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "SimSun",
    "Noto Sans CJK SC", "Source Han Sans SC",
    "WenQuanYi Zen Hei", "Arial Unicode MS", "PingFang SC",
]


def use_chinese_font():
    """挑一个系统里真实存在的 CJK 字体并应用。返回字体名，没找到返回 None。"""
    import matplotlib.font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False   # 负号别画成方块
            return name
    return None


def save(fig, name, dpi=150):
    """把图存到仓库根的 figures/ 下，返回路径。"""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [图] 已保存 {path.relative_to(ROOT)}")
    return path
