#!/usr/bin/env python3
"""检查页面的入口：一次跑完页面检查和内联脚本语法检查。

    python3 .dojo/scripts/site/check.py <页面...>
    python3 .dojo/scripts/site/check.py --all
    python3 .dojo/scripts/site/check.py --selftest

具体的检查在同目录各自的脚本里，也可以单独运行：
    check_page.py      页面
    check_script.py    内联脚本语法
    check_css.py       没人用的 CSS
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_css  # noqa: E402
import check_page  # noqa: E402
import check_script  # noqa: E402


def main(args: list[str]) -> int:
    if "--selftest" in args:
        codes = [m.selftest() for m in (check_page, check_script, check_css)]
        return 1 if any(codes) else 0

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

    failed = 0
    for page in pages:
        errors = check_page.check(page)
        if errors:
            failed += 1
            print(f"- {page}")
            for e in errors:
                print(f"  - {e}")

    script_errors, blocks, node_ok = check_script.check(pages)
    for e in script_errors:
        print(f"- {e}")
    if not node_ok:
        print("提示：PATH 里没有 node，跳过内联脚本语法检查")

    if failed or script_errors:
        print(f"检查失败：{failed}/{len(pages)} 个页面有问题" + (f"，{len(script_errors)} 块脚本有问题" if script_errors else ""))
        return 1
    print(f"检查通过：{len(pages)} 个页面" + (f"，内联脚本 {blocks} 块" if node_ok else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
