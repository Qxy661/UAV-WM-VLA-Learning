# -*- coding: utf-8 -*-
"""生成 README 首屏 hero 图与 GitHub 社交预览图。

两张图都只用本仓库的实测口径（见 C:/tmp/wm/wordcount.py 与 README 徽章），
不引入外部素材，跑一次即得，便于日后口径变化时重出。

用法:
    py -3.9 tools/make_brand_assets.py
输出:
    figures/hero.png            1600x560  README 首屏横幅
    figures/social-preview.png  1280x640  GitHub Settings -> Social preview 上传用
"""
import io
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.font_manager as fm

sys.stdout.reconfigure(encoding='utf-8')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# --- 中文字体：挑一个本机装了的 ---
_avail = {f.name for f in fm.fontManager.ttflist}
for _cand in ('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Source Han Sans SC'):
    if _cand in _avail:
        plt.rcParams['font.sans-serif'] = [_cand]
        FONT = _cand
        break
else:
    raise SystemExit('no CJK font found')
plt.rcParams['axes.unicode_minus'] = False

BG = '#0D1117'
CARD = '#161B22'
BORDER = '#30363D'
WHITE = '#F0F6FC'
MUTED = '#8B949E'
ACCENT = '#58A6FF'

TITLE = '世界模型 · VLA · VLM 学习库'
SUBTITLE = '面向无人机与具身智能 —— 从认知到前沿，每个结论都能重跑'
STATS = [
    ('59', '篇学习文档'),
    ('195', '篇论文 · 逐条核 arXiv 号'),
    ('24', '个纯 CPU 可跑 demo'),
    ('28.7万', '字正文'),
]


def _stat_boxes(ax, x0, x1, y0, h, gap, num_fs, lab_fs):
    n = len(STATS)
    w = (x1 - x0 - gap * (n - 1)) / n
    for i, (num, lab) in enumerate(STATS):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch(
            (x, y0), w, h,
            boxstyle='round,pad=0,rounding_size=14',
            linewidth=1.4, edgecolor=BORDER, facecolor=CARD))
        ax.text(x + w / 2, y0 + h * 0.60, num,
                ha='center', va='center', fontsize=num_fs,
                color=ACCENT, fontweight='bold')
        ax.text(x + w / 2, y0 + h * 0.24, lab,
                ha='center', va='center', fontsize=lab_fs, color=MUTED)


def make_hero(path):
    W, H = 1600, 560
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')

    ax.add_patch(Rectangle((0, 0), W, H, facecolor=BG, edgecolor='none'))
    # 左上角一道强调色，避免整块死板
    ax.add_patch(Rectangle((0, H - 8), W, 8, facecolor=ACCENT, edgecolor='none'))

    ax.text(W / 2, 452, TITLE, ha='center', va='center',
            fontsize=46, color=WHITE, fontweight='bold')
    ax.text(W / 2, 388, SUBTITLE, ha='center', va='center',
            fontsize=20, color=MUTED)

    _stat_boxes(ax, 110, W - 110, y0=118, h=150, gap=40, num_fs=36, lab_fs=15)

    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def make_social(path):
    W, H = 1280, 640
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')

    ax.add_patch(Rectangle((0, 0), W, H, facecolor=BG, edgecolor='none'))
    ax.add_patch(Rectangle((0, H - 10), W, 10, facecolor=ACCENT, edgecolor='none'))

    ax.text(W / 2, 512, TITLE, ha='center', va='center',
            fontsize=44, color=WHITE, fontweight='bold')
    ax.text(W / 2, 448, SUBTITLE, ha='center', va='center',
            fontsize=19, color=MUTED)

    _stat_boxes(ax, 90, W - 90, y0=196, h=150, gap=34, num_fs=32, lab_fs=14)

    ax.text(W / 2, 96, 'github.com/Qxy661/UAV-WM-VLA-Learning',
            ha='center', va='center', fontsize=17, color=ACCENT)

    fig.savefig(path, facecolor=BG)
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    make_hero('figures/hero.png')
    make_social('figures/social-preview.png')
    for p in ('figures/hero.png', 'figures/social-preview.png'):
        print('%s  %.1f KB' % (p, os.path.getsize(p) / 1024.0))
    print('font =', FONT)
