# -*- coding: utf-8 -*-
"""检查全仓文档里的相对链接是否都指向真实存在的文件/目录。

只查相对链接：外链（http/https/mailto）、纯锚点（#...）一律跳过。
指向目录的链接（以 / 结尾）也算通过，只要该目录存在——它们在 GitHub 与
文档站上都能落到该目录的 README/index。

用法:
    py -3.9 tools/check_links.py [--verbose]
退出码:
    0 全部存在；1 有缺失（逐条打印，含行号）
"""
import glob
import io
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

TARGETS = sorted(set(
    glob.glob('docs/**/*.md', recursive=True) +
    glob.glob('references/*.md') +
    glob.glob('mindmaps/*.md') +
    ['README.md', 'CONTRIBUTING.md', 'code/README.md']
))

MD_LINK = re.compile(r'!?\[[^\]]*\]\(\s*([^)\s]+)')
HTML_ATTR = re.compile(r'(?:src|href)="([^"]+)"')


def is_external(t):
    return ('://' in t or t.startswith('mailto:') or t.startswith('#')
            or t.startswith('data:'))


def check_file(path, verbose):
    """返回 (链接总数, [(行号, 原样链接, 解析后路径), ...])。"""
    total = 0
    bad = []
    with io.open(path, encoding='utf-8', errors='replace') as fh:
        for n, line in enumerate(fh, 1):
            for rx in (MD_LINK, HTML_ATTR):
                for m in rx.finditer(line):
                    t = m.group(1).strip()
                    if not t or is_external(t):
                        continue
                    t = t.split('#')[0].split('?')[0]
                    if not t:
                        continue
                    total += 1
                    resolved = os.path.normpath(
                        os.path.join(os.path.dirname(path), t))
                    exists = os.path.exists(resolved)
                    if not exists:
                        bad.append((n, t, resolved))
                    elif verbose:
                        print('  ok  %s:%d  %s' % (path, n, t))
    return total, bad


def main():
    verbose = '--verbose' in sys.argv
    total = 0
    bad = []
    for path in TARGETS:
        n_links, rows = check_file(path, verbose)
        total += n_links
        for n, t, _resolved in rows:
            bad.append((path, n, t))

    if bad:
        print('缺失的相对链接（%d）:' % len(bad))
        for path, n, t in bad:
            print('  %s:%d  ->  %s' % (path, n, t))
    print('%d 个文件、%d 条相对链接，缺失 %d 条' % (len(TARGETS), total, len(bad)))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
