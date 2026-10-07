# 脚本

在仓库根目录运行。

## 写概念页

    # 检查大纲
    python3 .dojo/scripts/concept/check_outline.py wiki/<name>/research/outline.md

    # 生成长页和短页，并自动检查正文；--run 同时运行文中的代码并核对输出
    python3 .dojo/scripts/concept/build_page.py wiki/<name>/research/content.md --run

    # 单独检查正文
    python3 .dojo/scripts/concept/check_content.py wiki/<name>/research/content.md

## 检查页面

    python3 .dojo/scripts/site/check.py wiki/<name>/index.html wiki/<name>/overview.html
    python3 .dojo/scripts/site/check.py --all

`check.py` 一次跑完页面检查和内联脚本语法检查。也可以单独运行：

    python3 .dojo/scripts/site/check_page.py <页面...>    # 页面
    python3 .dojo/scripts/site/check_script.py <页面...>  # 内联脚本语法，需要 node
    python3 .dojo/scripts/site/check_template.py          # 概念页模板
    python3 .dojo/scripts/site/check_css.py [--fix]       # 没人用的 CSS 规则

## 其他

    python3 .dojo/scripts/site/build_catalog.py --output _site/catalog.json

    # 改样式前后比对渲染，需要 Chrome
    python3 .dojo/scripts/site/diff_render.py <before-root> <after-root> <页面...>

## 自检

每个脚本都有 `--selftest`：

    python3 .dojo/scripts/concept/build_page.py --selftest
    python3 .dojo/scripts/concept/check_outline.py --selftest
    python3 .dojo/scripts/concept/check_content.py --selftest
    python3 .dojo/scripts/site/check.py --selftest     # 页面、脚本、模板、CSS 四个检查的自检

## 增删检查

`check_outline.py`、`check_content.py`、`check_page.py` 里，每条检查是一个函数，`RULES` 列出全部。增加检查就写一个函数加进 `RULES`，再往 `CASES` 加一个会触发它的坏输入；删除就两处一起删。
