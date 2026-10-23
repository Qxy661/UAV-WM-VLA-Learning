"""知识点（KP）注册表的校验与索引生成。

只依赖标准库：这台机器上没有 PyYAML，工具不应该为了读一张表引入依赖。

用法：
    python tools/kp.py derive      # 扫描 docs/，为每篇抽取 H2 候选知识点，写入 .work/metrics/kp-candidates.md
    python tools/kp.py check       # 校验 references/knowledge-points.md 与每篇文档头部
    python tools/kp.py stats       # 只打印统计

注册表格式（references/knowledge-points.md）：
    | ID | 标题 | 层 | 卷 | 承载文档 | 前置 | 判据 | 代码 | 状态 |

约束（check 会强制）：
    1. ID 唯一，形如 KP-001 或 KP-X01（X 表示横切判据）
    2. 层 ∈ {概念, 机制, 工程, 判据, 能力}
    3. 前置只能指向已存在且**不存在环**的 KP
    4. 判据列不得为空——没有判据的知识点不是知识点，是话题
    5. 每篇文档头部必须有 `本篇 KP`（可多行），列出的 ID 必须真实存在且承载文档包含本篇
    6. 文档路径必须存在（`待建` 开头的除外，计入 PENDING 而不是错误）
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
REGISTRY = ROOT / "references" / "knowledge-points.md"
OUT_DIR = ROOT / ".work" / "metrics"

LEVELS = {"概念", "机制", "工程", "判据", "能力"}
# 头部五字段（另保留既有的「预计阅读 | 前置知识」一行）。
# 「前置知识」里可以写 KP 编号，check 会校验它们是否存在——不再单设「前置」字段，
# 免得同一件事有两个名字。
CANON_KEYS = ["定位", "本篇 KP", "读完后应能", "判据", "预计阅读"]
STATUSES = {"完整", "待补", "待建", "争议"}
ID_RE = re.compile(r"^KP-(X\d{2}|\d{3})$")
SEP_CELL = re.compile(r"^:?-{2,}:?$")
HEADER_RE = re.compile(r"^>\s*\*\*(?P<key>[^*]+?)\*\*\s*[:：]\s*(?P<val>.+?)\s*$")
BOLD_LINE_RE = re.compile(r"^>\s*\*\*(?P<body>.+?)\*\*\s*$")


def parse_header_line(stripped: str) -> list[tuple[str, str]]:
    """解析三种既有写法，返回 (键, 值) 列表：
        `> 预计阅读：15 分钟 | 前置知识：基础深度学习概念`      （全库主流，无加粗）
        `> **预计阅读：15 分钟 | 前置知识：……**`               （整行加粗）
        `> **定位**：……`                                       （键值分写）
    """
    if not stripped.startswith(">"):
        return []
    body = stripped[1:].strip()
    wrapped = re.fullmatch(r"\*\*(.+)\*\*", body)
    if wrapped:
        body = wrapped.group(1)
    out: list[tuple[str, str]] = []
    for part in body.split("|"):
        part = part.strip().strip("*").strip()
        if not part:
            continue
        key, sep, val = part.partition("：")
        if not sep:
            key, sep, val = part.partition(":")
        key = key.strip().strip("*").strip()
        if sep and key and val.strip() and len(key) <= 12 and not any(c in key for c in "，。、；"):
            out.append((key, val.strip()))
    return out
LEVEL_HINTS = [
    ("判据", ("评测", "诊断", "度量", "指标", "口径", "判据", "检查", "基准", "误差")),
    ("工程", ("部署", "训练", "工程", "搭建", "复现", "优化", "实现", "环境", "导出")),
    ("机制", ("为什么", "机制", "原理", "如何工作", "因果", "本质", "权衡")),
]

FENCE = re.compile(r"^\s*(```|~~~)")

# Windows 控制台默认 cp936，输出中文会被上层按 UTF-8 解码成乱码；强制 UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


@dataclass
class Doc:
    rel: str
    path: Path
    header: dict[str, str] = field(default_factory=dict)
    sections: list[tuple[int, str]] = field(default_factory=list)
    lines: int = 0
    part: str = ""
    exempt: str = ""        # 非空表示豁免原因（文首注释 <!-- kp-exempt: 原因 -->）


def iter_docs() -> list[Doc]:
    out: list[Doc] = []
    for p in sorted(DOCS.rglob("*.md")):
        rel = p.relative_to(DOCS).as_posix()
        lines = p.read_text(encoding="utf-8").splitlines()
        d = Doc(rel=rel, path=p, lines=len(lines))
        d.part = rel.split("/")[0] if "/" in rel else "根"
        in_fence = False
        header_open = True
        for i, line in enumerate(lines, start=1):
            if FENCE.match(line):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            if header_open:
                stripped = line.strip()
                m = re.match(r"<!--\s*kp-exempt:\s*(.+?)\s*-->", stripped)
                if m and not d.exempt:
                    d.exempt = m.group(1)
                if stripped.startswith(">"):
                    for key, val in parse_header_line(stripped):
                        d.header.setdefault(key, val)
                elif stripped == "" or stripped.startswith("#"):
                    pass                      # 标题与引用块之间的空行不算头部结束
                else:
                    header_open = False       # 头部 = 文首连续引用块，遇到正文即结束
            if line.startswith("## ") and not line.startswith("### "):
                d.sections.append((i, line[3:].strip()))
        out.append(d)
    return out


def level_hint(title: str) -> str:
    for level, keys in LEVEL_HINTS:
        if any(k in title for k in keys):
            return level
    return "概念"


def cmd_derive(args) -> int:
    docs = iter_docs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total_sections = sum(len(d.sections) for d in docs)
    out = [
        "# 知识点候选清单（自动生成，勿手改）",
        "",
        f"- 文档数：{len(docs)}；H2 小节总数：{total_sections}",
        "- 候选 ≠ 知识点：一个 H2 可能合并成 1 个 KP，也可能一条都不算。",
        "- 层列是关键词猜的，只作提示。",
        "",
    ]
    for d in docs:
        out.append(f"## {d.rel}")
        out.append("")
        out.append(f"- 行数 {d.lines}；H2 {len(d.sections)}；现有头部："
                   f"{', '.join(d.header.keys()) if d.header else '（无）'}")
        out.append("")
        if not d.sections:
            out.append("（无 H2）")
            out.append("")
            continue
        out.append("| 行 | 候选标题 | 层提示 |")
        out.append("|---|---|---|")
        for ln, title in d.sections:
            out.append(f"| {ln} | {title} | {level_hint(title)} |")
        out.append("")
    dest = OUT_DIR / "kp-candidates.md"
    dest.write_text("\n".join(out), encoding="utf-8")

    per_part: dict[str, list[int]] = {}
    for d in docs:
        per_part.setdefault(d.part, []).append(len(d.sections))
    print(f"文档 {len(docs)} 篇，H2 {total_sections} 个 → {dest.relative_to(ROOT)}")
    for part in sorted(per_part):
        xs = per_part[part]
        print(f"  {part:<24} {len(xs):>2} 篇  H2 {sum(xs):>3} 个  平均 {sum(xs) / len(xs):.1f}")
    return 0


def parse_registry() -> tuple[list[dict[str, str]], list[str]]:
    if not REGISTRY.is_file():
        return [], [f"注册表不存在：{REGISTRY.relative_to(ROOT)}"]
    rows: list[dict[str, str]] = []
    errors: list[str] = []
    cols: list[str] | None = None
    in_table = False
    for raw in REGISTRY.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            in_table = False
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        # 分隔行：每格都是 --- / :---: / 空，不能只判断字符集合（否则 "-" 单元格的行会被误吞）
        if all(c == "" or SEP_CELL.match(c) for c in cells):
            in_table = True
            continue
        if not in_table:                      # 新表格的表头
            cols = cells if cells and cells[0].upper() == "ID" else None
            in_table = True
            continue
        if cols is None:
            continue
        if len(cells) != len(cols):
            errors.append(f"列数不符（期望 {len(cols)}，实际 {len(cells)}）：{line[:60]}…")
            continue
        rows.append(dict(zip(cols, cells)))
    return rows, errors


def cmd_check(args) -> int:
    rows, errors = parse_registry()
    ids = {r.get("ID", "") for r in rows}
    problems: list[str] = list(errors)

    for r in rows:
        rid = r.get("ID", "")
        if not ID_RE.match(rid):
            problems.append(f"ID 不合法：{rid!r}")
        if r.get("层") not in LEVELS:
            problems.append(f"{rid} 层不合法：{r.get('层')!r}")
        if r.get("状态") and r.get("状态") not in STATUSES:
            problems.append(f"{rid} 状态不合法：{r.get('状态')!r}")
        if not r.get("判据", "").strip() or r.get("判据") in {"-", "—", ""}:
            problems.append(f"{rid} 缺判据（没有判据的知识点不是知识点）")
        for pre in re.split(r"[,，、\s]+", r.get("前置", "")):
            if pre and pre not in {"-", "—", "无"} and pre not in ids:
                problems.append(f"{rid} 的前置 {pre} 不存在")
        doc = r.get("承载文档", "")
        if doc and not doc.startswith("待建"):
            # 多篇承载文档用「；」分隔：文件名本身可能带顿号（如 07-数据、预训练与跨具身.md）
            for piece in re.split(r"[；;\n]+", doc):
                if piece and not (DOCS / piece).is_file():
                    problems.append(f"{rid} 承载文档不存在：{piece}")

    # 环检测（前置图）
    graph = {r.get("ID", ""): [p for p in re.split(r"[,，、\s]+", r.get("前置", ""))
                               if p in ids] for r in rows}
    state: dict[str, int] = {}

    def dfs(n: str, stack: list[str]) -> None:
        state[n] = 1
        for m in graph.get(n, []):
            if state.get(m) == 1:
                problems.append("前置存在环：" + " → ".join(stack + [m]))
            elif state.get(m) is None:
                dfs(m, stack + [m])
        state[n] = 2

    for n in graph:
        if state.get(n) is None:
            dfs(n, [n])

    docs = iter_docs()
    covered: dict[str, list[str]] = {r.get("ID", ""): [] for r in rows}
    for r in rows:
        # 多篇承载文档用「；」分隔：文件名本身可能带顿号（如 07-数据、预训练与跨具身.md）
        for piece in re.split(r"[；;\n]+", r.get("承载文档", "")):
            if piece and not piece.startswith("待建"):
                covered.setdefault(r.get("ID", ""), []).append(piece)

    missing_header = 0
    exempted = 0
    bad_refs = 0
    KP_TOKEN = re.compile(r"KP-(?:X\d{2}|\d{3})")
    for d in docs:
        # 头部任何字段里出现的 KP 编号都必须真实存在（含「前置知识」里的跨篇前置）
        for key, val in d.header.items():
            for ref in KP_TOKEN.findall(val):
                if ref not in ids:
                    problems.append(f"{d.rel} 头部字段「{key}」引用了不存在的 {ref}")
                    bad_refs += 1
        if "本篇 KP" not in d.header:
            if d.exempt:
                exempted += 1
            else:
                missing_header += 1
            continue
        refs = [x for x in re.split(r"[,，、\s]+", d.header["本篇 KP"]) if x]
        for ref in refs:
            if ref not in ids:
                if not KP_TOKEN.fullmatch(ref):
                    problems.append(f"{d.rel} 的「本篇 KP」里有非 KP 编号：{ref}")
                bad_refs += 1
            elif d.rel not in covered.get(ref, []):
                problems.append(f"{d.rel} 头部声称承载 {ref}，但注册表未把本篇列入承载文档")
                bad_refs += 1

    pending = sum(1 for r in rows if r.get("承载文档", "").startswith("待建"))
    print(f"注册表：{len(rows)} 条 KP（待建 {pending}）；文档 {len(docs)} 篇")
    print(f"头部缺失「本篇 KP」：{missing_header}/{len(docs)} 篇（P1.3 待办）")
    if exempted:
        print(f"豁免五字段：{exempted} 篇（文首 <!-- kp-exempt: 原因 -->）")
    if problems:
        print(f"\n问题 {len(problems)} 条：")
        for p in problems[:40]:
            print("  -", p)
        if len(problems) > 40:
            print(f"  … 另有 {len(problems) - 40} 条")
        return 1
    print("校验通过。")
    return 0


def cmd_stats(args) -> int:
    docs = iter_docs()
    total = sum(d.lines for d in docs)
    secs = sum(len(d.sections) for d in docs)
    no_h2 = [d.rel for d in docs if not d.sections]
    exempt = [d.rel for d in docs if d.exempt]
    denom = len(docs) - len(exempt)
    print(f"文档 {len(docs)} 篇 / {total} 行 / H2 {secs} 个 / 平均 {total / len(docs):.0f} 行")
    print(f"无 H2 的文档：{no_h2 or '无'}")
    if exempt:
        print(f"豁免五字段：{len(exempt)} 篇（{', '.join(exempt)}）")
    keys: dict[str, int] = {}
    for d in docs:
        for k in d.header:
            keys[k] = keys.get(k, 0) + 1
    print(f"目标头部字段覆盖（P1.3 要把这五项做到 {denom}/{denom}）：")
    for k in CANON_KEYS:
        print(f"  {k:<8} {keys.get(k, 0):>2}/{denom}")
    other = {k: v for k, v in keys.items() if k not in CANON_KEYS}
    if other:
        print("其他既有字段：")
        for k, v in sorted(other.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<10} {v:>2}/{len(docs)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="KP 注册表工具（纯标准库）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("derive", help="抽取 H2 候选知识点")
    sub.add_parser("check", help="校验注册表与文档头部")
    sub.add_parser("stats", help="打印统计")
    args = ap.parse_args(argv)
    return {"derive": cmd_derive, "check": cmd_check, "stats": cmd_stats}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
