#!/usr/bin/env python3
"""所有 dojo:status=outdated 页面的校验：旧概念页，以及已暂停的论文、学习记录、数据流。

site/validate.py 对 dojo:status=outdated 的页面整页跳过。
这里把三个暂停类别的样式表补回去（旧概念页本来就在 validate.py 的表里），
并关掉跳过，用来在需要时校验那些旧页面：

    python3 .dojo/scripts/archive/legacy-validate.py <页面...>
    python3 .dojo/scripts/archive/legacy-validate.py --all-outdated

只做页面校验。CI 不跑。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "site"))
import page  # noqa: E402
import validate  # noqa: E402

# dataflow 有两代：旧页用 cytoscape + dojo-dataflow.css，新页用 ELK + dojo-flow.css。
page.TYPE_STYLESHEET.update({
    "paper": ("dojo-paper.css",),
    "note": ("dojo-note.css",),
    "dataflow": ("dojo-flow.css", "dojo-dataflow.css"),
})
page.SKIP_OUTDATED = False


def main() -> int:
    argv = sys.argv[1:]
    if "--all-outdated" in argv:
        argv = [a for a in argv if a != "--all-outdated"]
        argv += sorted(
            str(p) for p in Path("wiki").glob("*/index.html")
            if page.OUTDATED_RE.search(p.read_text(encoding="utf-8", errors="ignore"))
        )
    if not argv:
        print(__doc__)
        return 2
    sys.argv = [sys.argv[0]] + argv
    return validate.main()


if __name__ == "__main__":
    sys.exit(main())
