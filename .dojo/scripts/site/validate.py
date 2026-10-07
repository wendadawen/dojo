#!/usr/bin/env python3
"""发布闸门：页面、内联脚本、样式、首页目录，一次跑完。

站点部署前必须全绿的全部机械检查，入口只有一个。内容质量不在这里评
（那由独立审查负责）。

    # 页面校验（默认；顺带跑内联脚本语法，需要 node，缺失则跳过）
    python3 .dojo/scripts/site/validate.py <页面...>
    python3 .dojo/scripts/site/validate.py --all

    # 专项
    python3 .dojo/scripts/site/validate.py --templates
    python3 .dojo/scripts/site/validate.py --js <页面...>
    python3 .dojo/scripts/site/validate.py --css [--confirm] [--fix]
    python3 .dojo/scripts/site/validate.py --catalog --output _site/catalog.json

    # 改共享样式前后的渲染比对
    python3 .dojo/scripts/site/validate.py --diff <before-root> <after-root> <页面...>

校验逻辑按职责放在同目录：page.py（页面、模板、内联脚本）、styles.py（死 CSS、渲染比对）、
catalog.py（首页目录）、vocab.py（词表）。本文件只做调度。
"""

from __future__ import annotations

import concurrent.futures
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalog import run_catalog  # noqa: E402
from page import (  # noqa: E402
    TEMPLATE_DIR,
    TYPE_STYLESHEET,
    check_inline_js,
    check_template,
    is_outdated,
    validate_page,
)
from styles import run_css_check, run_render_diff  # noqa: E402


def main() -> int:
    argv = sys.argv[1:]

    if "--diff" in argv:
        i = argv.index("--diff")
        return run_render_diff(argv[i + 1:])
    if "--catalog" in argv:
        return run_catalog(argv)
    if "--css" in argv:
        return run_css_check(argv)
    if "--templates" in argv:
        template_root = Path(TEMPLATE_DIR)
        templates = sorted(template_root / kind / "index.html" for kind in TYPE_STYLESHEET)
        failures = [(t, check_template(t)) for t in templates]
        failures = [(t, errs) for t, errs in failures if errs]
        if failures:
            print("模板校验失败：")
            for template, errs in failures:
                print(f"- {template}")
                for error in errs:
                    print(f"  - {error}")
            return 1
        print(f"模板校验通过：{len(templates)} 个模板")
        return 0

    args = [a for a in argv if a not in {"--all", "--js"}]
    js_only = "--js" in argv
    if "--all" in argv:
        # research/ 不发布，里面的资料快照（抓取的网页等）不是站点页面
        args = sorted(str(p) for p in Path("wiki").glob("**/*.html") if "research" not in p.parts)
    if not args:
        print(__doc__)
        return 2

    pages: list[Path] = []
    missing: list[str] = []
    for arg in args:
        path = Path(arg)
        if path.is_dir():
            path = path / "index.html"
        if not path.exists():
            missing.append(str(path))
            continue
        pages.append(path)
    if missing:
        print("错误：页面不存在：" + "、".join(missing))
        return 2
    skipped = [p for p in pages if is_outdated(p)]
    if skipped:
        pages = [p for p in pages if p not in skipped]
        print(f"note: 跳过 {len(skipped)} 个 dojo:status=outdated 的页面")
        if not pages:
            return 0

    if js_only:
        js_errors, total, ran = check_inline_js(pages)
        if not ran:
            print("错误：PATH 里没有 node，这项检查需要安装 Node.js")
            return 2
        if js_errors:
            print(f"内联脚本检查失败：{total} 块里有 {len(js_errors)} 块有问题")
            for error in js_errors:
                print(f"- {error}")
            return 1
        print(f"内联脚本检查通过：{len(pages)} 个页面，{total} 块")
        return 0

    if len(pages) == 1:
        results = [(pages[0], validate_page(pages[0]))]
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda p: (p, validate_page(p)), pages))

    # 内联脚本语法：与逐页检查合并跑一次。没有 node 就跳过，只提示，
    # 不让缺一个可选工具导致整轮校验失败。
    js_errors, js_total, js_ran = check_inline_js(pages)
    if js_errors:
        results.append((Path("<内联脚本>"), js_errors))
    if not js_ran:
        print("note: 未找到 node，跳过内联脚本语法检查")

    failed = [(page, errors) for page, errors in results if errors]
    if failed:
        print(f"校验失败：{len(failed)}/{len(results)} 个页面")
        for page, errors in failed:
            print(f"- {page}")
            for error in errors:
                print(f"  - {error}")
        return 1

    suffix = f"，内联脚本 {js_total} 块" if js_ran else ""
    if len(pages) == 1:
        print(f"校验通过：{pages[0]}{suffix}")
    else:
        print(f"校验通过：{len(pages)} 个页面{suffix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
