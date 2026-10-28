"""把一集巡检录制成可看的回放页：逐帧画面 + 结构化曲线，单文件 HTML。

    python projects/vla-transfer/tools/replay.py --episode /root/work/data/uav96c/shard_00/episode_07200 \
        --out runs/replay_7200.html

设计取舍：
- 不依赖 matplotlib / ffmpeg / 浏览器插件，图像内嵌 base64，曲线用内联 SVG。
- 曲线画三条：到目档的距离、动作推力、画面里红色像素的占比。
  第三条是"到底看没看见目标"的直接观测，比任何注意力图都硬。
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np


def red_fraction(frame: np.ndarray) -> float:
    f = frame.astype(np.float32) / 255.0
    m = (f[..., 0] > 0.6) & (f[..., 0] > 1.8 * f[..., 1]) & (f[..., 0] > 1.8 * f[..., 2])
    return float(m.mean())


def svg_path(xs: list[float], ys: list[float], w: int, h: int, pad: int = 26) -> str:
    if not xs:
        return ""
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 - x0 < 1e-9:
        x1 = x0 + 1
    if y1 - y0 < 1e-9:
        y1 = y0 + 1
    pts = []
    for x, y in zip(xs, ys):
        px = pad + (x - x0) / (x1 - x0) * (w - 2 * pad)
        py = h - pad - (y - y0) / (y1 - y0) * (h - 2 * pad)
        pts.append(f"{px:.1f},{py:.1f}")
    return " ".join(pts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True, help="episode 目录（含 meta.json 与 obs.npz）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=120)
    a = ap.parse_args()

    ep = Path(a.episode)
    meta = json.loads((ep / "meta.json").read_text(encoding="utf-8"))
    z = np.load(ep / "obs.npz")
    imgs, ft = z["images"], z["frame_t"]
    pos, act, t = z["pos"], z["action"], z["t"]
    target = np.asarray(meta.get("target_pos", [0, 0, 0]), np.float32)
    dist = np.linalg.norm(pos - target[None], axis=1)

    step = max(1, len(imgs) // a.max_frames)
    frames = imgs[::step]
    frame_ft = ft[::step]

    W, H = 96, 96
    cells = []
    red_series = []
    for k, (fr, tt) in enumerate(zip(frames, frame_ft)):
        buf = io.BytesIO()
        try:
            from PIL import Image
            Image.fromarray(fr).save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            tag = f'<img src="data:image/png;base64,{b64}" width="{W}" height="{H}" loading="lazy">'
        except Exception:
            tag = f'<span class="noimg">frame {k}</span>'
        r = red_fraction(fr)
        red_series.append((float(tt), r))
        cells.append(f'<figure><figcaption>t={float(tt):.2f}s 红像素 {r*100:.2f}%'
                     f'</figcaption>{tag}</figure>')

    # 三条曲线：距离（米）、推力（N）、红占比（%）
    d_poly = svg_path(t.tolist(), dist.tolist(), 720, 140)
    thrust = act[:, 0] if act.size else np.zeros_like(t)
    th_poly = svg_path(t.tolist(), np.asarray(thrust).tolist(), 720, 140)
    red_poly = svg_path([p[0] for p in red_series], [p[1] * 100 for p in red_series], 720, 140)

    html = f"""<!doctype html><meta charset="utf-8">
<title>回放 {ep.name}</title>
<style>
 body{{font:14px/1.6 -apple-system,"Segoe UI",sans-serif;margin:24px;color:#111;background:#fafafa}}
 code{{background:#eee;padding:1px 4px;border-radius:3px}}
 .grid{{display:flex;flex-wrap:wrap;gap:8px}}
 figure{{margin:0;background:#fff;border:1px solid #ddd;padding:4px}}
 figcaption{{font-size:11px;color:#666}}
 svg{{background:#fff;border:1px solid #ddd}}
 h2{{font-size:15px;margin:22px 0 6px}}
 .kv{{display:grid;grid-template-columns:auto 1fr;gap:2px 14px;font-size:13px}}
</style>
<h1>回放 · {ep.name}</h1>
<div class="kv">
 <b>指令</b><span>{meta.get('instruction','')}</span>
 <b>布局</b><span>{meta.get('layout','a')}</span>
 <b>覆盖率</b><span>{meta.get('coverage','?')}</span>
 <b>目标位置</b><span>{[round(float(v),2) for v in target]}</span>
 <b>帧数 / 控制步</b><span>{len(imgs)} / {len(t)}</span>
</div>
<h2>到目标距离（米）</h2><svg width="720" height="140"><polyline fill="none" stroke="#0a7"
 stroke-width="1.6" points="{d_poly}"/></svg>
<h2>推力（N）</h2><svg width="720" height="140"><polyline fill="none" stroke="#a30"
 stroke-width="1.6" points="{th_poly}"/></svg>
<h2>画面红像素占比（%）</h2><svg width="720" height="140"><polyline fill="none" stroke="#c60"
 stroke-width="1.6" points="{red_poly}"/></svg>
<h2>逐帧（每 {step} 帧取 1）</h2>
<div class="grid">{''.join(cells)}</div>
"""
    Path(a.out).write_text(html, encoding="utf-8")
    print(json.dumps({"out": a.out, "frames": len(frames), "steps": len(t),
                      "dist_min": round(float(dist.min()), 3),
                      "red_max_pct": round(max(r for _, r in red_series) * 100, 2)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
