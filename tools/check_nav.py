# -*- coding: utf-8 -*-
"""核对 mkdocs.yml 的 nav 与仓库实际文档是否一致。

mkdocs 自己的 `validation.nav.omitted_files` 只能警告，且被 build_docs.py 生成的
卷落地页要特殊处理；这里做一次精确定义：

  1. 仓库里每一篇进站的 .md 都要在 nav 里出现（否则站上点不到）；
  2. nav 里每一条都要指向真实存在的文件（否则构建时静默少一页）；
  3. docs/<卷>/index.md 是构建期生成的落地页，必须有对应的 nav 条目。

nav 里的路径都相对 docs_dir（= build-docs/），而 build-docs/ 是仓库的镜像，
所以这些路径可以直接在仓库根下核对。

用法:
    py -3.9 tools/check_nav.py
退出码:
    0 一致；1 有不一致（缺项会逐条打印）
"""
import glob
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

MD = re.compile(r'[^\s:\'"]+\.md')


def expected_files():
    """仓库里应该出现在站点上的 .md（相对仓库根）。"""
    files = {'README.md', 'CONTRIBUTING.md'}
    for pat in ('docs/**/*.md', 'references/*.md', 'mindmaps/*.md', 'code/README.md'):
        files.update(glob.glob(pat, recursive=True))
    return {f.replace(os.sep, '/') for f in files}


def nav_entries():
    text = open('mkdocs.yml', encoding='utf-8').read()
    idx = text.find('\nnav:')
    if idx < 0:
        raise SystemExit('mkdocs.yml 里找不到 nav:')
    return set(MD.findall(text[idx:]))


def main():
    exp = expected_files()
    nav = nav_entries()
    generated = {e for e in nav if re.match(r'docs/[^/]+/index\.md$', e)}

    missing_from_nav = sorted(exp - nav)
    missing_on_disk = sorted(nav - exp - generated)
    orphan_landing = sorted(generated - {d + '/index.md' for d in
                                        {os.path.dirname(e) for e in exp}
                                        if d.startswith('docs/')})

    ok = True
    if missing_from_nav:
        ok = False
        print('文档没进 nav（%d）:' % len(missing_from_nav))
        for m in missing_from_nav:
            print('  -', m)
    if missing_on_disk:
        ok = False
        print('nav 指向不存在的文件（%d）:' % len(missing_on_disk))
        for m in missing_on_disk:
            print('  -', m)
    if orphan_landing:
        ok = False
        print('卷落地页指向不存在的卷目录（%d）:' % len(orphan_landing))
        for m in orphan_landing:
            print('  -', m)

    # 每个卷目录都应有落地页
    sections = sorted({os.path.dirname(e) for e in exp
                       if e.startswith('docs/') and os.path.dirname(e) != 'docs'})
    no_landing = [d for d in sections if d + '/index.md' not in nav]
    if no_landing:
        ok = False
        print('卷目录缺落地页 nav 条目（%d）:' % len(no_landing))
        for d in no_landing:
            print('  -', d + '/index.md')

    n_sec = len(sections)
    print('nav %d 条 / 仓库文档 %d 篇 / 卷落地页 %d 个 —— %s'
          % (len(nav), len(exp), n_sec, '一致' if ok else '不一致'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
