#!/usr/bin/env python3
"""已暂停类别（论文、学习记录、数据流）页面的校验。

site/validate.py 只维护现行的概念页，对 dojo:status=outdated 的页面整页跳过。
这里把三个暂停类别的样式表、概览文案补回去，并关掉跳过，用来在需要时校验那些旧页面：

    python3 .dojo/scripts/archive/legacy-validate.py <页面...>
    python3 .dojo/scripts/archive/legacy-validate.py --all-outdated

只做页面校验，不做模板、死 CSS、文案回写。CI 不跑。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("site_validate", HERE.parent / "site" / "validate.py")
validate = importlib.util.module_from_spec(spec)
sys.modules["site_validate"] = validate
spec.loader.exec_module(validate)

# dataflow 有两代：旧页用 cytoscape + dojo-dataflow.css，新页用 ELK + dojo-flow.css。
validate.TYPE_STYLESHEET.update({
    "paper": ("dojo-paper.css",),
    "note": ("dojo-note.css",),
    "dataflow": ("dojo-flow.css", "dojo-dataflow.css"),
})
validate.OVERVIEW_TERM["paper"] = "概览"
validate.DETAIL_TERM["paper"] = "完整解析"
validate.OVERVIEW_LABEL["paper"] = "论文概览"
validate.SKIP_OUTDATED = False


def main() -> int:
    argv = sys.argv[1:]
    if "--all-outdated" in argv:
        argv = [a for a in argv if a != "--all-outdated"]
        argv += sorted(
            str(p) for p in Path("wiki").glob("*/index.html")
            if validate.OUTDATED_RE.search(p.read_text(encoding="utf-8", errors="ignore"))
        )
    if not argv:
        print(__doc__)
        return 2
    sys.argv = [sys.argv[0]] + argv
    return validate.main()


if __name__ == "__main__":
    sys.exit(main())
