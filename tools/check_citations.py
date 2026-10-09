"""
check_citations.py — 全仓 arXiv 引用核查

扫描 docs/ 与 references/ 下的 Markdown，抽出所有 arXiv 号，逐条向 arXiv API
核对真实标题、作者、年份，把「文档声称的标题」与「API 返回的标题」不一致的
条目列出来。

用法：
    py -3.9 tools/check_citations.py            # 核查并打印摘要
    py -3.9 tools/check_citations.py --write    # 额外写出 references/citation-audit.md
    py -3.9 tools/check_citations.py --verbose  # 打印每一条的核对结果

三个必须守住的点（都是踩过的坑）：

  1. 一次请求可以带多个 id（`id_list=a,b,c`），但必须同时把 `max_results`
     设够。默认只回 10 条，超出的会被静默截断，看起来就像「这些论文不存在」。
  2. arXiv 会限流，返回的是 503 的 HTML 而不是 Atom。判据是响应里有没有
     `<feed>`：没有就是请求失败，必须重试。**绝不能把请求失败当成论文不存在**，
     否则会得到一整批假的「引用造假」结论。
  3. 从上下文猜出来的「文档声称的标题」只是线索，不是判据。表格里的第一列、
     链接文字、引号里的字符串都可能是标题，也可能是描述。最终以 API 为准，
     相似度低只代表「值得人看一眼」。
  4. **条目式的标题认不出来，等于没查。** 清单层（`paper-list.md`、各卷「关键论文」
     节、`docs/06` 卡片）的条目写成 `- **[会议'年月] 名称** — *English Title*`，
     标题只在斜体里，号在下一行的徽章 URL 里。两头都要堵：不认斜体就大批
     「无标题可判」（实测 195 处降到 140 处就是它的功劳），不滤徽章就会被
     `[![arXiv](https://img.shields.io/…)](…)` 截出 `![arXiv` 这种「链接文字」，
     于是每个条目都报标题不符。另：`*Nature Communications*` 这类出处名也是斜体，
     收进来会把号判成配错（2602.00708、2405.07736 就这么被顶上去过），见 `is_venue`。

只依赖标准库。
"""

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 必须走 https：http 端点回 301，urllib 跟随重定向时在部分网络下会整批失败，
# 于是最老的那几个号（1707/1803/1811…）会被批量错记为「未能核实」。
API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}

BATCH = 25                 # 每批请求的 id 数
SLEEP = 4.0                # 批间隔，arXiv 建议 3 秒以上
RETRY = 4                  # 单批失败重试次数
SIM_FLOOR = 0.50           # 标题相似度低于此值才报出来
WEAK = {"nearby", "fallback"}   # 兜底抽来的候选串，只够进「待看」，不足以定罪

# 现代格式 2312.12345；老格式 cs/0703122
RE_ID = re.compile(r"\b(\d{4}\.\d{4,5})\b")
RE_OLD = re.compile(r"\b([a-z-]+(?:\.[A-Z]{2})?/\d{7})\b")
RE_LINK = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
RE_QUOTED = re.compile(r"[\"“]([^\"”]{8,200})[\"”]")
# 斜体标题：`- **[arXiv'26.04] MotionScape** — *Motion-Stratified UAV Video ...*`。
# 前后各一个 `*`，用环视把 `**加粗**` 排除掉（加粗内部的首字符就是 `*`，
# 从那里起步会撞上第二条环视）。
RE_ITALIC = re.compile(r"(?<!\*)\*([^*\n]{8,220})\*(?!\*)")
# 加粗标题：条目式里没有系统名的篇目把完整标题放在加粗位
# （`- **[NeurIPS'18.03] World Models** — *Ha & Schmidhuber* · ★★★`），
# 这类条目连斜体位都没有标题，不认加粗就会退到「整行」当标题，条条报不符。
RE_BOLD = re.compile(r"\*\*([^*\n]{4,220})\*\*")
# 条目式开头的 `[Venue'YY.MM]` 标签，与标题比相似度前要剥掉。
RE_TAG = re.compile(r"^\s*\[[^\]]{0,40}\]\s*")


# ---------------------------------------------------------------- 抽取

PARTICLE = {"van", "der", "de", "von", "la", "del", "di", "den", "ter"}
_W = r"(?:[A-Z][A-Za-z'’.-]+|van|der|de|von|la|del|di|den|ter)"
RE_AUTHOR = re.compile(rf"^(?:{_W}(?:[,\s]+|$)){{1,8}}(?:et al\.?|等)?$")


def looks_like_authors(t):
    """「Kipf, van der Pol, Welling」「Alonso et al.」这类作者串，不是标题。

    表格里标题与作者是相邻单元格，不剔除的话每一行都会被当成标题不符。
    """
    s = t.strip()
    if " et al" in s or s.endswith("等"):
        return True
    if not RE_AUTHOR.match(s.replace(";", ",")):
        return False
    # 全大写词、有连字符的复合词都可能是模型名，别误杀
    words = [w for w in re.split(r"[,\s]+", s) if w]
    return all(w.lower() in PARTICLE or (w[:1].isupper() and len(w) <= 12) for w in words)


VENUE_WORD = {
    "nature", "science", "ieee", "acm", "cell", "pnas", "arxiv", "elife",
    "transactions", "journal", "proceedings", "letters", "magazine",
    "robotics", "automation", "communications", "on", "and", "the", "of", "for",
}


def is_venue(t):
    """「Nature Communications」「Science Robotics」这类出处名，不是标题。

    本仓的斜体有两种用法：标标题（`*"..."*` 与条目式的 `*English Title*`），
    以及标出处（表格里的 `*Science Robotics*`、正文里的 `*Nature Communications*`）。
    后者收进来只会把号判成配错——实测就是它把 2602.00708、2405.07736 两个号
    顶上了「对不上」名单，而原文其实写对了。

    判据取「**整串都由出处词组成**」而不是「以 Nature 开头」：后者会把
    「Nature-Inspired ...」这种真标题一并误杀。
    """
    s = t.strip().strip("*_\"“”")
    if re.search(r"子刊|期刊|学报|会议|出版社|预印本", s):
        return True
    w = [x.lower() for x in re.findall(r"[A-Za-z][A-Za-z-]*", s)]
    return bool(w) and len(w) <= 4 and all(x in VENUE_WORD for x in w)


def plausible(t, src):
    """判断一个候选串像不像「论文标题」。

    这一层专治误报：从散文、本地链接、作者单元格里抽出来的串如果不滤掉，
    会被当成「文档声称的标题」，于是每一行都报成不符。宁可漏，不可滥。
    """
    lo = 4 if src in ("table", "link", "quote", "prefix", "italic") else 10
    if not (lo <= len(t) <= 160):
        return False
    # 徽章行 `[![arXiv](https://img.shields.io/badge/arXiv-2604.07991-b31b1b.svg)](...)`
    # 会被 RE_LINK 截出 `![arXiv` 这个「链接文字」——它不是标题，而且因为它
    # 含号（号在 shields 的 URL 里），会让每个条目都报成标题不符。清单层
    # 改条目式后每个条目都带徽章，这一条不滤掉就是满屏误报。
    if "shields.io" in t or "[!" in t:
        return False
    if not re.search(r"[A-Za-z一-鿿]", t):               # 纯数字/标点，如年份单元格
        return False
    if re.search(r"[。！？；]", t):                       # 句末标点 = 散文
        return False
    if t.rstrip().endswith(("：", ":", "，", ",", "、")):  # 冒号逗号收尾 = 引子
        return False
    if re.search(r"\.md\b|^docs/|^code/|^figures/|^https?://", t):
        return False
    # 「arXiv」「arXiv:2401.1」「arxiv.org」这类纯标签，表格里是列头不是标题
    if re.fullmatch(r"(?i)(www\.)?arxiv(\.org)?\s*[:：]?\s*[\d.]*", t.strip()):
        return False
    if re.match(r"^\d{4}\s*年|^\d+\.\d+\s*$", t):         # 「2025 年 4 月」这类
        return False
    # 作者串在表格列、裸 URL 前缀、整行兜底里一律剔除；但 link / quote / italic /
    # bold 是**明确的标题位**——指向这个号的链接文字、引号里的话、条目式的斜体与
    # 加粗位，本身就是标题声明。这里不剔，否则短标题会被作者判据误杀：实测
    # `Gaussian Splatting SLAM`（三个词、全大写首字母）被判成作者串，于是
    # 2312.06741 那条被报成「对不上」，而文档写的正是论文原标题。
    if src not in ("link", "quote", "italic", "bold") and looks_like_authors(t):
        return False
    if is_venue(t):
        return False
    # 中文散文：汉字多且没有英文词
    if len(re.findall(r"[一-鿿]", t)) > 6 and not re.search(r"[A-Za-z]{3}", t):
        return False
    return True


def claimed_titles(lines, i, aid):
    """从第 i 行及其上文猜「文档声称的标题」，返回 [(文本, 来源)]。

    来源区分可信度：link/quote/table/prefix 是明确的标题位，nearby/fallback
    只是兜底。一行里出现多个 arXiv 号时（参考文献列表那种），兜底一律不用，
    否则会把 A 的标题配到 B 的号上，凭空造出不符。
    """
    out = []
    line = lines[i]
    multi = len(set(RE_ID.findall(line)) | set(RE_OLD.findall(line))) > 1

    def clean(pairs):
        """去掉强调号与多余空白，再滤掉不像标题的串。

        必须先滤再判「有没有候选」：`[2201.04339](...)` 的链接文字就是号
        本身，不滤掉的话它会被当成一个候选，于是上面那行的真标题没机会被
        取到，而它自己的相似度是 0，正好把号判成配错。
        """
        seen, uniq = set(), []
        for t, src in pairs:
            t = re.sub(r"\s+", " ", t.strip(" *_")).strip()
            if t and plausible(t, src) and t.lower() not in seen:
                seen.add(t.lower())
                uniq.append((t, src))
        return uniq

    for text, target in RE_LINK.findall(line):            # 链接文字
        if aid in target and text.strip():
            out.append((text.strip(), "link"))

    # **论文：** *"..."*。括号里的引号是出处不是标题：
    # `（NeurIPS 2021 "Physical Reasoning ..." workshop）` 这种，收进来就会
    # 把号判成配错，还挡住了往上看真标题的机会。
    spans = [m.span() for m in re.finditer(r"[（(][^）)]*[）)]", line)]
    for m in RE_QUOTED.finditer(line):
        if any(a <= m.start() < b for a, b in spans):
            continue
        out.append((m.group(1), "quote"))
    if not multi:                       # 一行多个号时，斜体分不清属于谁
        for m in RE_ITALIC.finditer(line):                # 斜体标题
            if any(a <= m.start() < b for a, b in spans):
                continue
            out.append((m.group(1), "italic"))

    # 必须在 clean 前判定，且判据是「有没有指向这个号的链接」而不是「链接文字像不像
    # 标题」。表格行 `| ... | [2604.02241](url) | ... |` 的链接文字就是号本身，
    # 按后者判会把它划进「裸 URL」分支，于是把整行前半截当成标题、凭空报不符。
    # 徽章行不受影响：`[![arXiv](https://img.shields.io/...)](arxiv.org/abs/号)`
    # 确实有链接，此处为真、跳过裸 URL 分支；而它的链接文字会被上面滤掉，clean 后
    # `out` 为空，照样落到「往上看两行」去取上一行的斜体标题。
    has_link = any(s == "link" for _, s in out)
    out = clean(out)

    # 裸 URL（无链接文字）时，取 URL 前面那段当标签，如「- DDPM 论文: <url>」。
    # 必须先确认真有这么个 URL：匹配不上时会取到整行，而表格行本身就以号开头，
    # 那样取到的「标签」是整行，任何节略写法都会被判成不符。
    if not has_link:
        m = re.search(r"https?://\S*" + re.escape(aid), line)
        if m:
            pre = re.sub(r"\[[^\]]*\]\($", "", line[:m.start()])
            pre = pre.strip(" -*_|#>\t")
            if pre:
                out.append((pre, "prefix"))
                out = clean(out)

    if line.lstrip().startswith("|"):                     # 表格行
        # 只取前两个不含号的单元格。本仓表格的列序是「号 | 标题/简称 | 主题 |
        # 出处 | 状态」，后面的列是「敏捷飞行」「*Science Robotics* 2026」这类
        # 主题与出处，当成标题去比只会误报。
        cells = [c.strip() for c in line.strip().strip("|").split("|")
                 if c.strip() and not RE_ID.search(c) and not c.startswith(":")]
        out = clean(out + [(c, "table") for c in cells[:2]])

    if not out and not multi:                             # 往上看两行
        # 只在当前行没有强候选时才上看。「**论文：** *"标题"*」与「**arXiv：**
        # 号」常分占两行，标题行不给当前行留线索（链接文字就是号本身，已被
        # clean 滤掉），所以要上看；但无条件上看会把邻行另一篇论文的标题算到
        # 本行的号头上，凭空造出不符。
        for j in range(i - 1, max(-1, i - 3), -1):
            prev = lines[j]
            # 上一行若带着**别的**号，它的标题属于那篇，不是本篇。原先只挡
            # 「一行多个号」，于是「一行一个号」的文献列表（`2405.07736  [主题] 标题`
            # 这种）会把上一条的标题算到本条的号头上——本条自己那行才是对的。
            ids_prev = set(RE_ID.findall(prev)) | set(RE_OLD.findall(prev))
            if any(x != aid for x in ids_prev):
                break
            # 条目式的标题行：`- **[arXiv'26.04] MotionScape** — *Motion-Stratified UAV ...*`。
            # 徽章与标题常分占两行，号在徽章那行，标题在上一行——斜体与引号都是
            # 明确的标题位，优先取；两个都取不到才退到「整行」。
            cand = [(m.group(1), "quote") for m in RE_QUOTED.finditer(prev)]
            cand += [(m.group(1), "italic") for m in RE_ITALIC.finditer(prev)]
            # 条目式里没有系统名的篇目，完整标题在加粗位而不是斜体位；
            # 先把开头的 `[Venue'YY.MM]` 标签剥掉再比。
            cand += [(RE_TAG.sub("", m.group(1)), "bold")
                     for m in RE_BOLD.finditer(prev)]
            if cand:
                out = clean(cand)
                if out:
                    break
            if prev.strip() and \
                    not prev.lstrip().startswith(("|", "#", "---")):
                out = clean([(prev.strip(), "nearby")])
                break

    if not out and not multi:                             # 兜底：整行
        t = RE_ID.sub("", line)
        t = re.sub(r"(?i)arxiv[:：]?\s*", "", t).strip(" *_|—-\t")
        if len(t) >= 10:
            out = clean([(t, "fallback")])

    return out


def extract():
    """扫描全仓 Markdown，返回 [{'file','line','id','claimed'}]。"""
    found = []
    files = sorted(list((ROOT / "docs").rglob("*.md")) + list((ROOT / "references").rglob("*.md")))
    files += [ROOT / "README.md", ROOT / "CONTRIBUTING.md"]
    for p in files:
        if not p.exists():
            continue
        rel = p.relative_to(ROOT).as_posix()
        if rel == "references/citation-audit.md":      # 本工具自己的产物，跳过
            continue
        lines = p.read_text(encoding="utf-8", errors="replace").split("\n")
        for i, line in enumerate(lines):
            for m in set(RE_ID.findall(line)) | set(RE_OLD.findall(line)):
                found.append({"file": rel, "line": i + 1, "id": m,
                              "claimed": claimed_titles(lines, i, m)})
    return found


# ---------------------------------------------------------------- 核对

def fetch(ids):
    """批量取元数据。返回 {id: {...}}。请求失败重试，失败返回 None 而不是空 dict。"""
    out = {}
    for k in range(0, len(ids), BATCH):
        chunk = ids[k:k + BATCH]
        url = f"{API}?id_list={','.join(chunk)}&max_results={len(chunk)}"
        body = None
        for attempt in range(RETRY):
            try:
                with urllib.request.urlopen(url, timeout=90) as r:
                    body = r.read().decode("utf-8", "replace")
            except Exception as e:                      # noqa: BLE001
                print(f"    请求异常({e.__class__.__name__})，重试 {attempt + 1}/{RETRY}", file=sys.stderr)
                time.sleep(SLEEP * (attempt + 1))
                continue
            if "<feed" in body:                         # 有效 Atom 响应
                break
            # 503 之类：拿到 HTML 而不是 Atom，必须重试而不是当成「查无此文」
            print(f"    非 Atom 响应（多半是限流），重试 {attempt + 1}/{RETRY}", file=sys.stderr)
            body = None
            time.sleep(SLEEP * (attempt + 1))
        if body is None:
            for c in chunk:
                out[c] = None                           # None = 未核实，不是「不存在」
            continue

        try:
            root = ET.fromstring(body)
        except ET.ParseError:
            for c in chunk:
                out[c] = None
            continue

        got = set()
        for e in root.findall("a:entry", NS):
            raw = (e.findtext("a:id", "", NS) or "").rsplit("/abs/", 1)[-1]
            aid = re.sub(r"v\d+$", "", raw)
            got.add(aid)
            out[aid] = {
                "title": " ".join((e.findtext("a:title", "", NS) or "").split()),
                "authors": [a.findtext("a:name", "", NS) for a in e.findall("a:author", NS)],
                "published": (e.findtext("a:published", "", NS) or "")[:10],
                "comment": " ".join((e.findtext("{http://arxiv.org/schemas/atom}comment", "", NS) or "").split()),
            }
        for c in chunk:
            if c not in got:
                out.setdefault(c, {})                   # {} = API 明确没返回这一条
        time.sleep(SLEEP)
    return out


def norm(s):
    s = s.lower()
    s = re.sub(r"^[^a-z0-9一-鿿]+", "", s)
    return re.sub(r"[^a-z0-9一-鿿]+", " ", s).strip()


def is_short_label(t):
    """「IRIS」「DDPM 论文」这类简称。它们可能指代正确却不含标题词，
    靠字符串比对判不了，单独归类交人看，不算不符。"""
    return len(t) <= 20 and len(re.findall(r"[A-Za-z]+", t)) <= 3


def sim(a, b):
    """文档声称的标题与 API 真实标题的相符程度。

    用**召回**而非 Jaccard：要回答的是「文档写的那些词，在真实标题里找得到
    吗」，不是「两个串长度像不像」。文档常用节略（`Precise Aggressive
    Maneuvers` / `Think Like a Pilot（FLIGHT）`）指代全称，Jaccard 会因为
    真实标题更长而把它们判成不符——那是最难查的一类误报。
    """
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return 0.0
    if na in nb or nb in na:
        return 0.95
    # 文档常拿简称指代论文（「DDPM 论文」「IRIS」「HDVIO」）。只认真正的
    # 模型名式 token：全大写、或非首字母有大写、或含数字。首字母大写不算，
    # 否则标题里的「Learning」会命中一大片。
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", a):
        if len(tok) < 4:
            continue
        modelish = tok.isupper() or any(c.isupper() for c in tok[1:]) or any(c.isdigit() for c in tok)
        if modelish and norm(tok) and norm(tok) in nb:
            return 0.80
    ta, tb = na.split(), set(nb.split())
    if not ta:
        return 0.0
    return sum(1 for w in ta if w in tb) / len(ta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="写出 references/citation-audit.md")
    ap.add_argument("--verbose", action="store_true", help="逐条打印")
    ap.add_argument("--fail-on-bad", action="store_true",
                    help="有 arXiv 号配错或 API 查无此文时退出码 1（供 CI 当门禁）；"
                         "请求失败的号只警告、不判失败")
    args = ap.parse_args()

    print("扫描 Markdown …")
    cites = extract()
    ids = sorted({c["id"] for c in cites})
    print(f"  命中引用 {len(cites)} 处，去重后 {len(ids)} 个 arXiv 号")
    print(f"分 {(len(ids) + BATCH - 1) // BATCH} 批请求 API（每批 {BATCH} 个，间隔 {SLEEP}s）…")

    meta = fetch(ids)
    unknown = sorted(i for i in ids if meta.get(i) is None)
    missing = sorted(i for i in ids if meta.get(i) == {})
    print(f"  核实 {len(ids) - len(unknown) - len(missing)} 个")
    if missing:
        print(f"  API 明确查无此文：{len(missing)} 个 -> {missing}")
    if unknown:
        print(f"  未能核实（请求失败，不能当作不存在）：{len(unknown)} 个 -> {unknown}")

    # 逐处比对。没有抽出任何像标题的候选 -> 判不了，不计入不符（避免误报）。
    mismatches, unverifiable, unjudged = [], [], []
    for c in cites:
        m = meta.get(c["id"])
        if m is None:
            unverifiable.append(c)
            continue
        if m == {}:
            c["real"], c["scores"] = None, []
            mismatches.append(c)
            continue
        real = m["title"]
        c["real"], c["meta"] = real, m
        c["scores"] = [{"claimed": t, "src": s, "sim": round(sim(t, real), 3)}
                       for t, s in c["claimed"]]
        if not c["scores"]:
            unjudged.append(c)
        elif any(s["sim"] >= SIM_FLOOR for s in c["scores"]):
            pass                                     # 对得上
        elif all(is_short_label(s["claimed"]) for s in c["scores"]):
            unjudged.append(c)                       # 简称，判不了，不算不符
        else:
            mismatches.append(c)

    # 同一个 (文件, 号) 只报一次
    seen, uniq = set(), []
    for c in mismatches:
        if (c["file"], c["id"]) not in seen:
            seen.add((c["file"], c["id"]))
            uniq.append(c)
    mismatches = sorted(uniq, key=lambda c: (c["file"], c["line"]))

    # 按 arXiv 号聚合：同一个号在全仓被配过哪些标题，最像的那个像不像
    by_id = {}
    for c in cites:
        m = meta.get(c["id"])
        if not m:
            continue
        e = by_id.setdefault(c["id"], {"real": m["title"], "best": None,
                                       "places": [], "titles": set()})
        e["places"].append(f"{c['file']}:{c['line']}")
        e["titles"].update(t for t, _ in c["claimed"])
        for t, src in c["claimed"]:
            if is_short_label(t):                    # 简称判不了，不参与判定
                continue
            if src in WEAK:
                continue        # 兜底抽来的串不足以定罪，只配进「待看」
            if len(re.findall(r"[一-鿿]", t)) >= 5:
                continue        # 整句中文是「描述」，不是「标题」：本仓表格里
                                # 两列并存，机器分不出译文与描述，宁可放过
            s = sim(t, m["title"])
            if e["best"] is None or s > e["best"]:
                e["best"] = s
    # 只有「有像标题的候选、且全都不像」才算号配错了
    bad_ids = {k: v for k, v in by_id.items()
               if v["best"] is not None and v["best"] < SIM_FLOOR}

    print(f"\n结论：{len(bad_ids)} 个 arXiv 号对不上任何一处文档标题；"
          f"{len(unjudged)} 处无标题可判（不计）")
    if unknown:
        # 请求失败的号不参与任何判定，也就不会出现在结论里。不显式报出来，
        # 一批失败就会静默少查几十个号，看起来却像「全查过了」。
        print(f"注意：{len(unknown)} 个号请求失败、未能核实（不代表不存在），"
              f"它们是：{'、'.join(sorted(unknown))}")
    for aid, v in sorted(bad_ids.items()):
        print(f"\n  arXiv {aid}  （出现 {len(v['places'])} 处）")
        for t in sorted(v["titles"])[:3]:
            print(f"    文档写：{t}")
        print(f"    实际是：{v['real']}")

    if args.verbose:
        print("\n---- 全部核对明细 ----")
        for c in cites:
            m = meta.get(c["id"])
            tag = "未核实" if m is None else ("查无" if m == {} else
                                        ("无标题可判" if not c["scores"] else
                                         ("OK" if any(s["sim"] >= SIM_FLOOR for s in c["scores"]) else "不符")))
            print(f"  [{tag}] {c['file']}:{c['line']} {c['id']}")

    if args.write:
        out = ROOT / "references" / "citation-audit.md"
        L = ["# arXiv 引用核查报告", "",
             "> 由 `tools/check_citations.py` 自动生成，请勿手改。",
             "> 判定以 arXiv API 返回的元数据为准。",
             "> 「未能核实」表示请求失败，**不代表论文不存在**，重跑即可。",
             "> 「无标题可判」表示文档没在 arXiv 号附近写出标题，本工具无从比对，不等于正确。",
             "> **本报告只校验两件事：号能否解析、号附近的标题是否配得上。**",
             "> 它**不校验正文数字、机制名、能力断言与数据集规模**——「0 个对不上」",
             "> 不能当作正文内容正确的证据。数字与机制必须回原文核。",
             "", "## 摘要", "",
             f"- 引用出现处：{len(cites)}",
             f"- 去重后 arXiv 号：{len(ids)}",
             f"- **对不上任何处文档标题的号：{len(bad_ids)}**",
             f"- API 查无此文：{len(missing)}",
             f"- 未能核实（请求失败）：{len(unknown)}",
             f"- 无标题可判的引用处：{len(unjudged)}", ""]

        # 章节号按实际写出的节递增，避免全为 0 时报告从「四」开头
        CN = "一二三四五六七八九十"
        sec = [0]

        def head(title):
            sec[0] += 1
            return [f"## {CN[sec[0] - 1]}、{title}", ""]

        if bad_ids:
            L += head("arXiv 号与文档标题对不上") + [
                  "按号聚合。同一个号在全仓被配过的所有标题里，最像的也达不到阈值，",
                  "说明这个号多半配错了。", "",
                  "| arXiv | 实际是 | 文档写的是 | 出现处 |", "|---|---|---|---|"]
            for aid, v in sorted(bad_ids.items()):
                t = "<br>".join(sorted(v["titles"])[:3])
                where = "<br>".join(f"`{p}`" for p in v["places"][:4])
                if len(v["places"]) > 4:
                    where += f"<br>…另 {len(v['places']) - 4} 处"
                L.append(f"| {aid} | {v['real']} | {t} | {where} |")
            L.append("")

        if missing:
            L += head("API 查无此文") + ["| 文件:行 | arXiv |", "|---|---|"]
            L += [f"| `{c['file']}:{c['line']}` | {c['id']} |"
                  for c in cites if meta.get(c["id"]) == {}]
            L.append("")

        if unknown:
            L += head("未能核实（请求失败，需重跑）") + [
                  "这些不是「不存在」，只是本次请求被限流或超时。重跑本脚本即可。", "",
                  "| arXiv |", "|---|"] + [f"| {i} |" for i in unknown] + [""]

        L += head("无标题可判的引用处") + [
              "文档给了号但附近没写标题，本工具判不了。要把它们变成可判的，",
              "在号附近补上标题即可。", "",
              "| 文件:行 | arXiv |", "|---|---|"]
        L += [f"| `{c['file']}:{c['line']}` | {c['id']} |" for c in unjudged[:200]]
        if len(unjudged) > 200:
            L.append(f"| … | 共 {len(unjudged)} 处 |")

        out.write_text("\n".join(L), encoding="utf-8")
        print(f"\n报告已写出：{out.relative_to(ROOT)}")

    # 退出码：只有「号配错」「API 明确查无」算失败。
    # 请求失败（unknown）绝不能当失败——脚本自己的 docstring 就警告过，
    # 把请求失败当成论文不存在会让整个结论作废。
    if args.fail_on_bad:
        if unknown:
            print(f"::warning::{len(unknown)} 个号请求失败、未能核实，不作为失败依据："
                  f"{'、'.join(sorted(unknown))}")
        if bad_ids or missing:
            print(f"\n--fail-on-bad：{len(bad_ids)} 个号配错、{len(missing)} 个号查无此文 -> 退出 1")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
