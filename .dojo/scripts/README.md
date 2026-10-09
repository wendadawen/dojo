# 脚本

在仓库根目录运行，具体命令见 `guides/concept.md`、`guides/dataflow.md` 和 `.github/workflows/pages.yml`。

## concept/

- `build_page.py`：读 `content.md`，生成长页和短页，`--run` 同时运行文中代码并核对输出。
- `check_outline.py`：检查大纲。
- `check_content.py`：检查正文。
- `common.py`：上面三个脚本共用的常量、依据检查和自检输出。

## dataflow/

- `build_page.py`：读 `dataflow.md`，生成 `index.html`。
- `check_source.py`：核查页面对不对得上源码——源码位置回查、节点名、权重形状，`--geometry` 量连线、`--shots` 截图。
- `common.py`：上面两个脚本共用的常量和自检输出。

## site/

- `check.py`：检查页面，页面规则加内联脚本语法（需要 node）；`--all` 查所有页面。
- `check_page.py`、`check_script.py`：`check.py` 调用的页面规则和脚本语法检查。
- `check_css.py`：找出共享样式表里没人用的规则，`--fix` 直接删除。
- `build_catalog.py`：扫描 `wiki/`，生成首页读的 `catalog.json`。
- `vocab.py`：主题和标签的封闭词表。

首页 `index.html` 引用的 `assets/` 和 `catalog.json` 只在 CI 里生成：`assets/` 来自 `.dojo/site/`，`catalog.json` 来自 `build_catalog.py`，不用手工收录页面。

`check.py`、`check_css.py` 和 concept、dataflow 下的脚本有 `--selftest`。增加检查：写一个函数加进 `RULES`，再往 `CASES` 加一个会触发它的坏输入；删除则两处一起删。
