## 这个 PR 做了什么

<!-- 一两句话说清楚，不用复述 diff -->

## 自查清单

- [ ] 改动只落在声明过的范围内，没顺手动别的地方
- [ ] 新增/修改的相对链接都能打开：`python tools/check_links.py`
- [ ] 新增文档已加进 `mkdocs.yml` 的 nav：`python tools/check_nav.py`
- [ ] 新增/改动的 arXiv 号来自官方：`python tools/check_citations.py`（只校验号与标题，正文数字仍需回原文核）
- [ ] 文中的数字来自实跑，不是估的；量不出来的地方写明「不作为结论」
- [ ] 新增的 `code/` 脚本只用 numpy + matplotlib，纯 CPU 可跑、两分钟内出图
- [ ] 控制台输出没有非 ASCII 字符（避免 GBK 终端报错）

## 涉及的文件

<!-- 主要改了哪几个 -->
