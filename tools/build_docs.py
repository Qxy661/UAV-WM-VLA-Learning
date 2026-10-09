# -*- coding: utf-8 -*-
"""把参与站点的目录按原相对层级镜像到 build-docs/，供 mkdocs 使用。

为什么要镜像而不是直接 `docs_dir: .`：仓库根有 1.1 GB 的 paper/，而 `mkdocs serve`
不遵守 exclude_docs，直接在根目录起站会把本地 serve 拖垮。镜像到一棵干净的子树，
既让 paper/ 进不来，又保持每个文件与仓库里的相对位置完全一致——于是仓库里既有的
相对链接（如 docs/xx.md 里的 ../../references/yy.md）在站点上指向同一个位置，
**一条链接都不用改写**。

用法:
    py -3.9 tools/build_docs.py
输出:
    build-docs/   （已在 .gitignore 中）
"""
import os
import shutil
import sys

sys.stdout.reconfigure(encoding='utf-8')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'build-docs')
ENTRIES = [
    'README.md', 'README.en.md', 'CONTRIBUTING.md', 'CHANGELOG.md', 'CITATION.cff',
    'docs', 'references', 'mindmaps', 'figures', 'code', 'tools',
]
IGNORE = shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo')


def count_md(path):
    n = 0
    for _root, _dirs, files in os.walk(path):
        n += sum(1 for f in files if f.endswith('.md'))
    return n


def _h1(path):
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            if line.startswith('# '):
                return line[2:].strip()
    return os.path.splitext(os.path.basename(path))[0]


def _section_title(dirname):
    # '01-基础概念' -> '基础概念'
    return dirname.split('-', 1)[1] if '-' in dirname else dirname


def make_section_index(dirpath, title):
    """为卷目录生成落地页，让 README / 正文里的 `docs/NN-xxx/` 目录链接在站点上能打开。

    只写进 build-docs/，仓库正文一字不动。内容机械取自子文件的 H1，不做任何撰写。
    """
    children = sorted(f for f in os.listdir(dirpath)
                      if f.endswith('.md') and f not in ('index.md', 'README.md'))
    if not children:
        return None
    lines = ['# %s' % title, '', '本节共 %d 篇。' % len(children), '']
    for f in children:
        label = _h1(os.path.join(dirpath, f))
        lines.append('- [%s](%s)' % (label, f))
    lines.append('')
    out = os.path.join(dirpath, 'index.md')
    with open(out, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(lines))
    return out


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)

    skipped = []
    for e in ENTRIES:
        src = os.path.join(ROOT, e)
        dst = os.path.join(OUT, e)
        if not os.path.exists(src):
            skipped.append(e)
            continue
        if os.path.isdir(src):
            shutil.copytree(src, dst, ignore=IGNORE)
        else:
            shutil.copy2(src, dst)

    # 卷目录落地页
    docs_out = os.path.join(OUT, 'docs')
    made = 0
    for d in sorted(os.listdir(docs_out)):
        dp = os.path.join(docs_out, d)
        if os.path.isdir(dp) and not os.path.exists(os.path.join(dp, 'index.md')):
            if make_section_index(dp, _section_title(d)):
                made += 1

    n_md = count_md(OUT)
    n_files = sum(len(f) for _r, _d, f in os.walk(OUT))
    print('build-docs/ 已生成: %d 个文件，其中 %d 篇 .md（含 %d 个卷落地页）'
          % (n_files, n_md, made))
    if skipped:
        print('跳过（仓库里不存在）:', ', '.join(skipped))


if __name__ == '__main__':
    sys.exit(main())
