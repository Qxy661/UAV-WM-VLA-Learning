# -*- coding: utf-8 -*-
"""维护 docs/en/index.md（英文文档总览）——负责防漂移，不覆盖手写内容。

英文门面只做门面：正文保持中文，这一页给英文读者一个能看懂的目录。
页面里的英文标题与说明是人写的，所以本脚本【不会】重写它；它只做两件事：

    py -3.9 tools/gen_en_index.py --write     初次或大改时，打印一份骨架到 stdout
    py -3.9 tools/gen_en_index.py --check     校验每个卷目录 / 每篇文档都被总览页覆盖

改了文档的标题或结构（增删文件、改所在卷），跑 --check；缺了谁它会点名。
只改正文不必管。

用法:
    py -3.9 tools/gen_en_index.py [--check|--write]
退出码:
    0 覆盖完整；1 有文档没被总览页收录
"""
import glob
import io
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

EN_INDEX = 'docs/en/index.md'
MD_LINK = re.compile(r'\]\(([^)\s]+\.md)(?:#[^)]*)?\)')


def repo_docs():
    """应被英文总览覆盖的文档（相对仓库根），不含英文总览自身。

    根级的四个门面文件（中英首页、贡献指南、demo 说明）也在预期集里：总览页的抬头与
    页脚本来就要指回它们；放进预期集，--check 才能把"指向了别的野路径"也当成错误。
    """
    out = ['README.md', 'README.en.md', 'CONTRIBUTING.md', 'code/README.md']
    for pat in ('docs/**/*.md', 'mindmaps/*.md', 'references/*.md'):
        for f in glob.glob(pat, recursive=True):
            f = f.replace(os.sep, '/')
            if f == EN_INDEX:
                continue
            out.append(f)
    return sorted(set(out))


def covered_docs():
    if not os.path.exists(EN_INDEX):
        return None
    text = io.open(EN_INDEX, encoding='utf-8').read()
    base = os.path.dirname(EN_INDEX)
    got = set()
    for m in MD_LINK.finditer(text):
        t = m.group(1)
        if '://' in t:
            continue
        got.add(os.path.normpath(os.path.join(base, t)).replace(os.sep, '/'))
    return got


def h1(path):
    with io.open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            if line.startswith('# '):
                return line[2:].strip()
    return os.path.basename(path)


def write_skeleton():
    """把当前文档树摊成骨架，供人工补英文说明。"""
    cur = None
    for f in repo_docs():
        sec = os.path.dirname(f)
        if sec != cur:
            cur = sec
            print('\n### %s\n' % (sec or '.'))
        rel = os.path.relpath(f, 'docs/en').replace(os.sep, '/')
        print('- **[English title TBD]** — one-line English description.  ')
        print('  `%s` → [%s](../%s)' % (h1(f), h1(f), rel))


def check():
    exp = set(repo_docs())
    got = covered_docs()
    if got is None:
        print('%s 不存在' % EN_INDEX)
        return 1
    missing = sorted(exp - got)
    extra = sorted(got - exp)
    if missing:
        print('英文总览页没收录（%d）:' % len(missing))
        for m in missing:
            print('  -', m)
    if extra:
        print('英文总览页指向了不该有的路径（%d）:' % len(extra))
        for e in extra:
            print('  -', e)
    print('待覆盖 %d 篇 / 已覆盖 %d 篇 —— %s'
          % (len(exp), len(exp & got), '完整' if not missing and not extra else '不完整'))
    return 1 if (missing or extra) else 0


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else '--check'
    if mode == '--write':
        write_skeleton()
        return 0
    if mode == '--check':
        return check()
    print(__doc__)
    return 2


if __name__ == '__main__':
    sys.exit(main())
