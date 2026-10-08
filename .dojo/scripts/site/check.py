#!/usr/bin/env python3
"""检查页面：页面规则加内联脚本语法（需要 node）。

    python3 .dojo/scripts/site/check.py <页面或目录...>
    python3 .dojo/scripts/site/check.py --all
    python3 .dojo/scripts/site/check.py --selftest
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "concept"))
import check_page  # noqa: E402
import check_script  # noqa: E402
from common import report  # noqa: E402


def selftest() -> int:
    results = [("页面：" + n, ok) for n, ok in check_page.selftest()]
    if shutil.which("node"):
        results += [("脚本：" + n, ok) for n, ok in check_script.selftest()]
    return report(results, "页面检查")


def main(args: list[str]) -> int:
    if "--selftest" in args:
        return selftest()
    if "--all" in args:
        # research/ 不发布，里面的资料快照不是站点页面
        pages = sorted(p for p in Path("wiki").glob("**/*.html") if "research" not in p.parts)
    else:
        pages = [Path(a) / "index.html" if Path(a).is_dir() else Path(a) for a in args]
    if not pages:
        print(__doc__)
        return 2
    missing = [str(p) for p in pages if not p.exists()]
    if missing:
        print("错误：页面不存在：" + "、".join(missing))
        return 2

    errors = []
    for page in pages:
        errors += [f"{page}：{e}" for e in check_page.check(page)]
    blocks = 0
    if shutil.which("node"):
        script_errors, blocks = check_script.check(pages)
        errors += script_errors
    else:
        print("提示：PATH 里没有 node，跳过内联脚本语法检查")
    for e in errors:
        print(f"- {e}")
    if errors:
        print(f"检查失败：{len(errors)} 个问题")
        return 1
    print(f"检查通过：{len(pages)} 个页面，内联脚本 {blocks} 块")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
