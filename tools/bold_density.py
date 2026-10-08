# -*- coding: utf-8 -*-
"""
bold_density.py — 正文段落加粗密度实测

`CONTRIBUTING.md` 的「排版」一节给了一条可核对的加粗规则，并附一个实测数字。
本脚本就是产出那个数字的口径，防止它随文档增删悄悄过期。

口径（写死在这里，改口径必须同时改 CONTRIBUTING）：

  * 语料：`docs/` 下全部 `*.md`。
  * 只统计**正文段落**：跳过围栏代码块、标题、表格行、列表项、引用块、
    分隔线、HTML 标签行（`<details>` 等）与空行；其余连续多行视为一个段落。
  * 「一处」= 段落里一个完整的 `**…**`。同一段落内多个各算一处。
  * 其中**标签式**单列：位于段落开头、且紧接 `：` 或直接结束的那一处
    （`**一句话总结**`、`**推荐阅读顺序**：`）。规则本就豁免"卡片字段标签"，
    把它从密度里摘出来，剩下的才是段落内的术语与结论加粗。
  * 分母 = 正文段落里的汉字数（U+4E00–U+9FFF），不含标题、表格、代码、列表。
  * 引用块默认跳过：`> **判据**：` 与 `> **预计阅读**：` 是结构锚点，与表格首列同类。
    要看含引用块的数字，加 `--include-quote`。

两条数字都要看：**含标签**的密度衡量排版观感，**去标签**的密度才是这条规则管的对象。

用法：
    py -3.9 tools/bold_density.py                  # 打印逐篇表与汇总
    py -3.9 tools/bold_density.py --include-quote  # 把引用块也算进正文
    py -3.9 tools/bold_density.py --write          # 写出 references/bold-density.md
    py -3.9 tools/bold_density.py --top 10         # 只看加粗最多的 N 篇（0 = 全列）

只依赖标准库。控制台是 GBK 时请加 `-X utf8` 或重定向到文件。
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

RE_BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
RE_LABEL = re.compile(r"\A\s*\*\*([^*]+?)\*\*(?=[:：]|\s*\Z)", re.S)
RE_HANZI = re.compile(r"[一-鿿]")
RE_LIST = re.compile(r"^\s{0,3}([-*+]|\d+\.)\s")
RE_FENCE = re.compile(r"^\s{0,3}(```|~~~)")
RE_HR = re.compile(r"^\s{0,3}(-{3,}|\*{3,}|_{3,})\s*$")


def paragraphs(text, include_quote=False):
    """切出正文段落，返回段落字符串列表。"""
    out, buf, in_code = [], [], False
    for raw in text.split("\n"):
        line = raw.rstrip("\r")
        if RE_FENCE.match(line):
            in_code = not in_code
            if buf:
                out.append("\n".join(buf))
                buf = []
            continue
        if in_code:
            continue
        stripped = line.strip()
        skip = (
            not stripped
            or stripped.startswith("#")
            or stripped.startswith("|")
            or stripped.startswith("<")
            or RE_LIST.match(line)
            or RE_HR.match(line)
            or (stripped.startswith(">") and not include_quote)
        )
        if skip:
            if buf:
                out.append("\n".join(buf))
                buf = []
            continue
        buf.append(line)
    if buf:
        out.append("\n".join(buf))
    return out


def measure(path, include_quote=False):
    paras = paragraphs(path.read_text(encoding="utf-8"), include_quote)
    joined = "\n".join(paras)
    total = len(RE_BOLD.findall(joined))
    label = sum(1 for q in paras if RE_LABEL.match(q))
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "paragraphs": len(paras),
        "bold": total,
        "label": label,
        "inline": total - label,
        "hanzi": len(RE_HANZI.findall(joined)),
    }


def main():
    ap = argparse.ArgumentParser(description="正文段落加粗密度实测")
    ap.add_argument("--include-quote", action="store_true", help="把引用块也算作正文")
    ap.add_argument("--top", type=int, default=0, help="只列加粗最多的 N 篇（0 = 全部）")
    ap.add_argument("--write", action="store_true", help="写出 references/bold-density.md")
    args = ap.parse_args()

    rows = [measure(p, args.include_quote) for p in sorted(DOCS.rglob("*.md"))]
    for r in rows:
        r["d_all"] = (r["bold"] * 1000.0 / r["hanzi"]) if r["hanzi"] else 0.0
        r["d_inline"] = (r["inline"] * 1000.0 / r["hanzi"]) if r["hanzi"] else 0.0

    hanzi = sum(r["hanzi"] for r in rows)
    bold = sum(r["bold"] for r in rows)
    label = sum(r["label"] for r in rows)
    inline = bold - label
    zero = [r for r in rows if r["bold"] == 0]
    heavy = sorted([r for r in rows if r["inline"] > 15], key=lambda r: -r["inline"])

    ranked = sorted(rows, key=lambda r: (-r["inline"], r["path"]))
    shown = ranked if args.top <= 0 else ranked[: args.top]

    print(f"语料：docs/ 下 {len(rows)} 篇"
          f"{'（含引用块）' if args.include_quote else '（不含引用块）'}")
    print(f"正文段落 {sum(r['paragraphs'] for r in rows)} 段，汉字 {hanzi}")
    print(f"加粗 {bold} 处 = 含标签 {bold * 1000.0 / hanzi:.1f} 处/千汉字"
          f"；其中标签式 {label} 处，段内 {inline} 处 = {inline * 1000.0 / hanzi:.1f} 处/千汉字")
    print(f"0 处的 {len(zero)} 篇；段内加粗超过 15 处的 {len(heavy)} 篇\n")
    print(f"{'段内':>4}  {'标签':>4}  {'合计':>4}  {'千字':>6}  {'段内/千字':>8}  文档")
    for r in shown:
        print(f"{r['inline']:>4}  {r['label']:>4}  {r['bold']:>4}  "
              f"{r['hanzi']/1000.0:>6.1f}  {r['d_inline']:>8.1f}  {r['path']}")
    if heavy:
        print("\n段内加粗超过 15 处的：")
        for r in heavy:
            print(f"  {r['inline']:>3} 处 / {r['paragraphs']} 段  {r['path']}")

    if args.write:
        dst = ROOT / "references" / "bold-density.md"
        lines = [
            "# 正文段落加粗密度实测",
            "",
            "> 由 `py -3.9 tools/bold_density.py --write` 生成，请勿手改。",
            "> 口径见脚本 docstring，规则本身见 `CONTRIBUTING.md` 的「排版」一节。",
            "> 数字随语料变化，行内已写明本次语料规模。",
            "",
            f"语料：`docs/` 下 {len(rows)} 篇。正文段落 "
            f"{sum(r['paragraphs'] for r in rows)} 段，汉字 {hanzi}，加粗 {bold} 处"
            f"（含标签 **{bold * 1000.0 / hanzi:.1f} 处/千汉字**）。"
            f"其中标签式 {label} 处，规则豁免；段内加粗 {inline} 处"
            f"（**{inline * 1000.0 / hanzi:.1f} 处/千汉字**）才是这条规则管的对象。"
            f"加粗为 0 的 {len(zero)} 篇，段内加粗超过 15 处的 {len(heavy)} 篇。",
            "",
            "| 段内加粗 | 标签式 | 合计 | 正文汉字（千） | 段内/千汉字 | 文档 |",
            "|---:|---:|---:|---:|---:|---|",
        ]
        for r in ranked:
            lines.append(
                f"| {r['inline']} | {r['label']} | {r['bold']} | "
                f"{r['hanzi']/1000.0:.1f} | {r['d_inline']:.1f} | `{r['path']}` |"
            )
        dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n已写出 {dst.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
