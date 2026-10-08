"""
vla_watch.py — VLA 前沿增量追踪

按词表查 arXiv，把**最近窗口内新出现、且仓库还没收录**的论文列出来，写成
`references/vla-watch-<年-月>.md`。词表按 `docs/03-VLA专题/06`–`10` 五篇新文档分节，
外加一节无人机主线。目的不是"查新"（那是 `paper/literature/_tools/netq.py` 的活），
而是**让专题不再过期**：过几个月重跑一次，就知道要补哪些。

用法：
    py -3.9 tools/vla_watch.py                    # 最近 120 天，全部词表，只打印
    py -3.9 tools/vla_watch.py --days 30          # 缩小窗口
    py -3.9 tools/vla_watch.py --group 06         # 只跑某一节（06/07/08/09/10/uav）
    py -3.9 tools/vla_watch.py --write            # 写出 references/vla-watch-YYYY-MM.md
    py -3.9 tools/vla_watch.py --json out.json    # 另存机器可读结果
    py -3.9 tools/vla_watch.py --all              # 连已收录的也列出来（默认只列新增）

三个必须守住的点（都是这个仓库踩过的坑）：

  1. **429 / 超时 / 非 200 记 NULL，绝不记 0。** 把请求失败当成"没有命中"会
     凭空造出一个空白，整个追踪结论就废了。见 `paper/notes/gap-roadmap.md` 纪律 F。
  2. **TOTAL 不是"相关论文数"。** arXiv 的 `all:` 只覆盖题名/摘要/作者/注释，
     不覆盖全文；一条宽查询命中几百条是没信息的，所以本工具的表格只列**新增**条目，
     TOTAL 只用来判断"这个词有没有人用"。
  3. **0 命中的第一反应是怀疑查询。** 命中 0 时本工具会打 `[?]` 标记，提示换个提法
     或换索引复核，而不是当作"这里没人做"。

取数一律 subprocess 调 curl 且**不加 `-x`**：本机 Python 的 urllib 会静默读取
Windows 注册表里的系统代理，把请求送到 127.0.0.1:7897 再被对端限流成 429。
这条是 `paper/literature/_tools/netq.py` 实测出来的，此处照抄。

只依赖标准库。
"""

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom",
      "os": "http://a9.com/-/spec/opensearch/1.1/"}
UA = "uav-wm-vla-learning-vla-watch/1.0"

RETRY = 4          # 单条查询失败重试次数（照抄 check_citations.py）
SLEEP = 4.0        # 查询间隔，arXiv 建议 3 秒以上
PER_QUERY = 30     # 每条查询取回的条目数上限

# 末尾的 `(?:v\d+)?` 是必须的：paper/ 里存的是带版本号的 arXiv 链接
# （`.../abs/2606.11618v1`），而 `\b` 在 `11618v` 之间不成立，旧写法会**整条漏掉**。
# 用 `(?!\d)` 而不是 `\b` 收尾，既容忍 `vN` 又不把长数字串切成两半。
RE_ID = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)")

# 词表。每节对应一篇新文档；`uav` 节是全场主线，不单独成篇。
# 写法说明：用 abs: 而不是 all:，避开作者名/注释里的偶然命中。
QUERIES = {
    "06": ("动作头与动作分块", [
        'abs:"action chunking"',
        'abs:"action head" AND abs:"vision-language"',
        'abs:"flow matching" AND abs:"robot policy"',
        'abs:"diffusion policy" AND abs:"manipulation"',
        'abs:"action tokenization"',
        'abs:"receding horizon" AND abs:"imitation learning"',
    ]),
    "07": ("数据、预训练与跨具身", [
        'abs:"cross-embodiment"',
        'abs:"robot dataset" AND abs:"pretraining"',
        'abs:"co-training" AND abs:"robot"',
        'abs:"latent action" AND abs:"robot"',
        'abs:"Open X-Embodiment"',
        'abs:"human video" AND abs:"robot policy"',
    ]),
    "08": ("强化学习后训练与自我改进", [
        'abs:"reinforcement learning" AND abs:"vision-language-action"',
        'abs:"GRPO" AND abs:"robot"',
        'abs:"reward model" AND abs:"robot policy"',
        'abs:"self-improvement" AND abs:"robot"',
        'abs:"world model" AND abs:"policy improvement"',
        'abs:"online fine-tuning" AND abs:"robot policy"',
    ]),
    "09": ("评测基准与报告口径", [
        'abs:"LIBERO"',
        'abs:"SimplerEnv"',
        'abs:"VLA benchmark"',
        'abs:"real robot" AND abs:"evaluation protocol"',
        'abs:"success rate" AND abs:"robot policy" AND abs:"evaluation"',
        'abs:"reproducibility" AND abs:"robot learning"',
    ]),
    "10": ("世界模型增强 VLA", [
        'abs:"world model" AND abs:"vision-language-action"',
        'abs:"video prediction" AND abs:"robot policy"',
        'abs:"imagined rollout"',
        'abs:"latent dynamics" AND abs:"policy learning"',
        'abs:"model-based" AND abs:"vision-language-action"',
    ]),
    "uav": ("无人机主线", [
        'abs:"aerial" AND abs:"vision-language-action"',
        'abs:"UAV" AND abs:"vision-language-action"',
        'abs:"drone" AND abs:"vision-language navigation"',
        'abs:"aerial" AND abs:"language model" AND abs:"control"',
        'abs:"quadrotor" AND abs:"foundation model"',
        'abs:"aerial" AND abs:"embodied" AND abs:"policy"',
    ]),
}


# 命中 0 的查询，按纪律 F 复核过之后把结论钉在这里，随生成一起写出去。
# 不写这个表的话，复核结论只活在对话里，下次重跑又变成一个裸的 0。
VERIFIED_ZEROS = {
    'abs:"receding horizon" AND abs:"imitation learning"':
        "**复核过：这是词汇层面的假空白。**换提法 `abs:\"execution horizon\"` 命中 **22**、"
        "`abs:\"action horizon\"` 命中 **11**（同 120 天窗口）。"
        "「receding horizon」是控制论的说法，VLA 社区写「action / execution horizon」。"
        "**不得记录为「这个方向没人做」。**",
    'abs:"quadrotor" AND abs:"foundation model"':
        "**复核过：零落在术语上，不落在领域上。**"
        "`aerial` / `UAV` / `drone foundation model`、`aerial foundation models` "
        "四种提法命中全为 **0**（两套词表交集为空，够得上「事实级空白」的判据）；"
        "但同窗口 `abs:\"aerial\" AND abs:\"vision-language-action\"` 命中 **11**。"
        "结论只能写到这一步：**空中领域不用「foundation model」自我描述，而用 VLA/VLN**。"
        "**不得写成「空中没有基础模型工作」。**",
}


# ---------------------------------------------------------------- 取数

def curl(url, timeout=90):
    """返回 (http_code, body_bytes)。不加 -x，直连。"""
    p = subprocess.run(
        ["curl", "-s", "--max-time", str(timeout), "-A", UA,
         "-w", "\n__HTTP__%{http_code}", url],
        capture_output=True)
    if p.returncode != 0:
        return None, b""
    out = p.stdout
    marker = b"\n__HTTP__"
    i = out.rfind(marker)
    if i < 0:
        return None, out
    try:
        code = int(out[i + len(marker):].decode().strip())
    except ValueError:
        return None, out[:i]
    return code, out[:i]


def fetch(query, n=PER_QUERY):
    """单条查询。返回 dict 或 None（None 表示请求失败，调用方记 NULL）。

    只认 http 200 且能解析出 feed；其余一律 None。
    """
    url = API + "?" + urllib.parse.urlencode(
        {"search_query": query, "start": 0, "max_results": n,
         "sortBy": "submittedDate", "sortOrder": "descending"})
    for attempt in range(RETRY):
        code, body = curl(url)
        if code == 200:
            try:
                root = ET.fromstring(body)
            except ET.ParseError:
                time.sleep(SLEEP)
                continue
            t = root.find("os:totalResults", NS)
            total = int(t.text) if t is not None and t.text else None
            items = []
            for e in root.findall("a:entry", NS):
                items.append({
                    "id": e.find("a:id", NS).text.split("/abs/")[-1],
                    "date": (e.find("a:published", NS).text or "")[:10],
                    "title": " ".join(e.find("a:title", NS).text.split()),
                })
            return {"query": query, "total": total, "items": items}
        time.sleep(SLEEP)
    return None


# ---------------------------------------------------------------- 台账

def known_ids():
    """扫仓库里已收录的 arXiv 号。排除本工具自己的产出，否则会自我去重成空。

    2026-10-08 修：原先漏了 `paper/`，而那里有 4500+ 条 ID 的文献台账，导致
    484 条"新增"里有 86 条其实早已收录，头部的"仓库尚未收录"就是假的。
    凡是本仓库里写过的 ID 一律算已收录——宁可少报新增，也不制造假信号。
    """
    ids = set()
    targets = list((ROOT / "docs").rglob("*.md"))
    targets += list((ROOT / "references").glob("*.md"))
    targets += list((ROOT / "mindmaps").glob("*.md"))
    targets += list((ROOT / "paper").rglob("*.md"))
    targets += [ROOT / "README.md", ROOT / "CONTRIBUTING.md",
                ROOT / "code" / "README.md"]
    for p in targets:
        if not p.exists() or p.name.startswith("vla-watch"):
            continue
        try:
            ids |= set(RE_ID.findall(p.read_text(encoding="utf-8")))
        except OSError:
            continue
    return ids


# ---------------------------------------------------------------- 主流程

def run(groups, days, show_all):
    since = (date.today() - timedelta(days=days)).strftime("%Y%m%d")
    until = date.today().strftime("%Y%m%d")
    window = f"submittedDate:[{since}0000 TO {until}2359]"

    known = known_ids()
    results, n_q, n_fail, n_new = [], 0, 0, 0

    for gid, (gname, queries) in QUERIES.items():
        if groups and gid not in groups:
            continue
        block = []
        for q in queries:
            r = fetch(f"{q} AND {window}")
            n_q += 1
            if r is None:
                n_fail += 1
                print(f"  [NULL] {q}  <- 请求失败，记 NULL 不记 0")
                block.append({"query": q, "total": None, "new": []})
                time.sleep(SLEEP)
                continue
            new = [it for it in r["items"] if it["id"].split("v")[0] not in known]
            shown = r["items"] if show_all else new
            n_new += len(new)
            capped = r["total"] > PER_QUERY
            if r["total"] == 0:
                flag = "  [?] 命中 0，先怀疑查询，别当空白"
            elif capped:
                flag = f"  （只取了最近 {PER_QUERY} 条，新增数是下界）"
            else:
                flag = ""
            print(f"  {r['total']:>6}  新增 {len(new):>3}  {q}{flag}")
            block.append({"query": q, "total": r["total"], "new": shown,
                          "capped": capped})
            time.sleep(SLEEP)
        results.append({"id": gid, "name": gname, "queries": block})

    return {"window": window, "days": days, "n_queries": n_q,
            "n_failed": n_fail, "n_new": n_new, "groups": results}


def to_markdown(res):
    d = date.today().strftime("%Y-%m")
    since, until = res["window"].split("[")[1].split(" TO ")
    since = f"{since[:4]}-{since[4:6]}-{since[6:8]}"
    until = f"{until[:4]}-{until[4:6]}-{until[6:8]}"
    L = [
        f"# VLA 前沿增量 · {d}",
        "",
        f"> 生成：`py -3.9 tools/vla_watch.py --write`｜窗口 {since} ~ {until}"
        f"（{res['days']} 天）｜词表 {res['n_queries']} 条｜请求失败 {res['n_failed']} 条"
        f"｜新增 {res['n_new']} 条",
        "",
        "**读法**（纪律 F 的镜像规则）：TOTAL 是**题摘层面的命中数，不是相关论文数**。"
        "一条宽查询命中几百条不含任何信息，所以下表只列**仓库尚未收录**的条目。"
        "`NULL` 表示请求失败（429/超时），**不是 0 命中**；`0` 表示确实没有，"
        "但第一反应应当是怀疑查询措辞，换个提法复核后再下结论。",
        "",
        "arXiv 的 `all:` / `abs:` 只覆盖题名/摘要/作者/注释，**不覆盖全文**。",
        "",
        "**去重范围是整仓**：`docs/`、`references/`、`mindmaps/`、`paper/` 下的 `.md`，"
        "加 `README.md` / `CONTRIBUTING.md` / `code/README.md`。"
        "`paper/` 下的 `_work*/` 取数缓存（json/xml/html/pdf）**不算收录**，故不计入——"
        "那里放的是原始查询结果，不是读过的文献。"
        "所以某条 ID 不在本表，只能推出**整仓没写过**，推不出\"VLA 专题没有\"。",
        "（2026-10-08 修：`paper/` 原先漏扫，加上 ID 正则不吃 `v1` 版本号后缀，"
        "两条合起来让 27 条早已在 `paper/` 笔记里读过的论文被标成\"新增\"。）",
        "",
    ]
    for g in res["groups"]:
        L += [f"## {g['id']} · {g['name']}", ""]
        for b in g["queries"]:
            if b["total"] is None:
                L += [f"**`{b['query']}`** — NULL（请求失败，**不是 0**）", ""]
                continue
            if not b["new"]:
                if b["total"] == 0:
                    note = VERIFIED_ZEROS.get(b["query"])
                    if note:
                        L += [f"**`{b['query']}`** — 命中 0。{note}", ""]
                    else:
                        L += [f"**`{b['query']}`** — 命中 0。"
                              "**先怀疑查询措辞，别当空白**：换个提法、换套词表"
                              "（或换索引）复核后再下结论——本仓库已经实测出两个假空白。", ""]
                    continue
                L += [f"**`{b['query']}`** — 命中 {b['total']}，无新增", ""]
                continue
            cap = (f"（只取了最近 {PER_QUERY} 条，新增数是**下界**）"
                   if b.get("capped") else "")
            L += [f"**`{b['query']}`** — 命中 {b['total']}，其中新增 {len(b['new'])}{cap}",
                  "", "| 日期 | arXiv | 标题 |", "|---|---|---|"]
            for it in b["new"]:
                L.append(f"| {it['date']} | `{it['id']}` | {it['title'][:110]} |")
            L.append("")
    L += ["---", "",
          "*本文件由 `tools/vla_watch.py` 生成，重跑即可刷新。"
          "收录进正文前，ID 与标题一律以 arXiv API 为准。*", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="VLA 前沿增量追踪")
    ap.add_argument("--days", type=int, default=120, help="回溯天数，默认 120")
    ap.add_argument("--group", action="append",
                    help="只跑指定节（06/07/08/09/10/uav），可重复")
    ap.add_argument("--write", action="store_true", help="写出 references/vla-watch-YYYY-MM.md")
    ap.add_argument("--json", help="另存机器可读 JSON")
    ap.add_argument("--all", action="store_true", help="已收录的也列出来")
    args = ap.parse_args()

    unknown = set(args.group or []) - set(QUERIES)
    if unknown:
        sys.exit(f"未知分组：{sorted(unknown)}；可选 {sorted(QUERIES)}")

    print(f"窗口：最近 {args.days} 天｜分组：{args.group or '全部'}")
    res = run(args.group, args.days, args.all)
    print(f"\n完成：{res['n_queries']} 条查询，失败 {res['n_failed']}，新增 {res['n_new']}")

    if args.write:
        out = ROOT / "references" / f"vla-watch-{date.today().strftime('%Y-%m')}.md"
        out.write_text(to_markdown(res), encoding="utf-8")
        print(f"已写出 {out.relative_to(ROOT)}")
    if args.json:
        Path(args.json).write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写出 {args.json}")


if __name__ == "__main__":
    main()
