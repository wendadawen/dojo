# 脚本

在仓库根目录运行。

## 写概念页

    # 检查大纲
    python3 .dojo/scripts/concept/check_outline.py wiki/<name>/research/outline.md

    # 生成长页和短页，并自动检查正文；--run 同时运行文中的代码并核对输出
    python3 .dojo/scripts/concept/build.py wiki/<name>/research/content.md --run

    # 单独检查正文
    python3 .dojo/scripts/concept/check_content.py wiki/<name>/research/content.md

## 校验页面

    python3 .dojo/scripts/site/validate.py wiki/<name>/index.html wiki/<name>/overview.html
    python3 .dojo/scripts/site/validate.py --all

## 其他

    python3 .dojo/scripts/site/validate.py --templates   # 检查模板
    python3 .dojo/scripts/site/validate.py --css [--fix] # 找没人用的 CSS 规则
    python3 .dojo/scripts/site/validate.py --catalog --output _site/catalog.json

    # 改样式前后比对渲染，需要 Chrome
    python3 .dojo/scripts/site/validate.py --diff <before-root> <after-root> <页面...>

## 自检

    python3 .dojo/scripts/concept/build.py --selftest
    python3 .dojo/scripts/concept/check_outline.py --selftest
    python3 .dojo/scripts/concept/check_content.py --selftest

## 增删检查

`check_outline.py` 和 `check_content.py` 里，每条检查是一个函数，文件末尾的 `RULES` 列出全部。增加检查就写一个函数加进 `RULES`，再往文件里的 `CASES` 加一个会触发它的坏输入；删除就两处一起删。
