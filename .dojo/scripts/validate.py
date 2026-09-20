#!/usr/bin/env python3
"""Validate wiki pages (index.html / overview.html) or the page templates.

Deterministic checks only: shell integrity, template leftovers, duplicate
ids, same-page anchors and broken local references. Semantic quality is out
of scope (handled by the independent review).

Usage:
    python3 .dojo/scripts/validate.py wiki/<name>/index.html
    python3 .dojo/scripts/validate.py wiki/*/index.html wiki/*/overview.html
    python3 .dojo/scripts/validate.py --all
    python3 .dojo/scripts/validate.py --templates

一次进程可校验多个文件，避免逐页起进程的开销。
"""

from __future__ import annotations

import concurrent.futures
import re
import sys
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

from catalog_builder import ALLOWED_TAGS, ALLOWED_TOPICS, ALLOWED_TYPES


LOCAL_ASSET_RE = re.compile(r'''(?:href|src)=["']([^"']+)["']''')
ID_RE = re.compile(r"""\bid=["']([^"']+)["']""")
SAME_PAGE_ANCHOR_RE = re.compile(r"""href=["']#([^"']*)["']""")
PLACEHOLDER_RE = re.compile(r"【[^】]*】")

IGNORE_PREFIXES = (
    "http://",
    "https://",
    "data:",
    "#",
    "mailto:",
)

MARKERS = ("@content", "@component", "TODO", "TBD")
REQUIRED_WIKI_META = (
    "description",
    "dojo:summary",
    "dojo:type",
    "dojo:topics",
    "dojo:tag",
)
# 每个 dojo:type 对应唯一的共享样式表，防止页面套错模板后仍能通过校验。
TYPE_STYLESHEET = {
    "concept": "dojo-concept.css",
    "paper": "dojo-paper.css",
    "note": "dojo-note.css",
    "dataflow": "dojo-dataflow.css",
}
TEMPLATE_DIR = ".dojo/templates"
STYLE_RE = re.compile(r"<style[^>]*>(.*?)</style>", re.S | re.I)

# research/ 允许的文件名集合（四类页面共用）。目录只放 .md，扩写记录与
# 审查轮次都按固定名字编号；sources/ 与 official/ 存放引文原始快照。
RESEARCH_ALLOWED_FILES = {
    "scope.md",
    "evidence.md",
    "outline.md",
    "glossary.md",
    "prereq-audit.md",
    "draft-check.md",
    "minor-fixes.md",
    "check-status.md",
    "arbitration.md",
    "measured.md",
}
RESEARCH_ALLOWED_DIRS = {"sources", "official"}
REVIEW_FILE_RE = re.compile(r"review-\d+\.md")
INLINE_MATH_RE = re.compile(
    r"\$\$[\s\S]+?\$\$|"
    r"(?<!\\)\$(?!\$)(?:\\.|[^$\n])+?(?<!\\)\$(?!\$)|"
    r"\\\([\s\S]+?\\\)|"
    r"\\\[[\s\S]+?\\\]"
)

# 数学符号必须写成 LaTeX 交给 KaTeX 渲染，不能直接使用 Unicode 字符。
# 只收录必须由 KaTeX 排版的字符：希腊字母、上下标、数学运算符与关系符。
# × – → 等在中文技术散文中作为普通排版字符使用，不列入。
BARE_MATH_CHARS = (
    "\u0391-\u03a9\u03b1-\u03c9"  # 希腊字母大小写
    "\u2202\u2207\u221a\u221d\u221e\u2211\u220f\u222b"  # 偏导 梯度 根号 正比 无穷 求和 求积 积分
    "\u2248\u2260\u2261\u2264\u2265\u226a\u226b"  # 约等 不等 恒等 小于等于 大于等于 远小于 远大于
    "\u2208\u2209\u2282\u2283\u2229\u222a\u2205"  # 属于 不属于 子集 超集 交 并 空集
    "\u2295\u2297\u22c5"  # 直和 张量积 点乘
    "\u2070-\u209f"  # 上标与下标数字字母
)
BARE_MATH_RE = re.compile(f"[{BARE_MATH_CHARS}]")

# 界面元素中的字符属于交互控件，不参与公式检查。
UI_CONTEXT_TAGS = {"button", "option", "title", "nav"}

# SVG 的 <text> 由浏览器直接绘制，KaTeX 不会处理其中的 $...$。
# 图内需要公式时改用 <foreignObject> 承载 HTML，KaTeX 可正常渲染其内容。
SVG_TEXT_RE = re.compile(r"<(text|tspan)\b[^>]*>(.*?)</\1>", re.S)

# 用 ASCII 近似写数学符号同样属于未渲染公式：
# 下标式标识（R_1、theta_i）、写成单词的希腊字母与角度单位、乘除号的字符替代。
ASCII_MATH_PATTERNS = (
    re.compile(r"\b[A-Za-z]\w*_\{?[A-Za-z0-9]"),
    re.compile(
        r"\b(?:alpha|beta|gamma|delta|theta|lambda|sigma|omega|phi|psi|mu|tau|epsilon|rho)\b",
        re.I,
    ),
    re.compile(r"\b(?:deg|degrees?|pi)\b", re.I),
    re.compile(r"[0-9A-Za-z]\s*\*\s*[0-9A-Za-z]"),
)

# 正文以文字形式提到的 research/ 文件必须真实存在。
# 实测产物与来源快照不随站点发布（CI 用 --exclude='research/' 排除整个目录），
# 正文里指向它们的路径在线上必然是 404。只有 measured.md 是明确允许的登记文件，
# 指南要求正文写「实测得到」并指向它，所以单独放行。
RESEARCH_MENTION_RE = re.compile(r"research/[\w./-]*[\w/-]")
RESEARCH_ALLOWED_RE = re.compile(r"^research/measured\.md$")

# 结构图使用 HTML 或内联 SVG，不使用等宽字符拼出的框线图。
BOX_DRAWING_RE = re.compile(r"[\u2500-\u257f\u2580-\u259f\u25a0-\u25ff\u2b00-\u2bff]")
PRE_BLOCK_RE = re.compile(r"<pre\b[^>]*>([\s\S]*?)</pre>", re.IGNORECASE)
BOX_DRAWING_MIN_HITS = 4

VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}


def strip_math_segments(text: str) -> str:
    """移除 $...$、$$...$$、\\(...\\)、\\[...\\] 包裹的公式内容。"""
    return INLINE_MATH_RE.sub(" ", text)


class PageInspector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.scripts: list[str] = []
        self.stylesheets: list[str] = []
        self.visible_parts: list[str] = []
        self.prose_parts: list[tuple[str, str]] = []
        self._ignored_depth = 0
        self._code_depth = 0
        self._tag_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple]) -> None:
        values = dict(attrs)
        if tag in {"script", "style"}:
            self._ignored_depth += 1
        if tag in {"code", "pre", "samp", "kbd"}:
            self._code_depth += 1
        if tag not in VOID_TAGS:
            self._tag_stack.append(tag)
        if tag == "meta":
            name = values.get("name", "").strip().lower()
            if name:
                self.meta[name] = values.get("content", "").strip()
        elif tag == "script" and values.get("src"):
            self.scripts.append(values["src"])
        elif tag == "link" and values.get("rel") == "stylesheet":
            self.stylesheets.append(values.get("href", ""))

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1
        if tag in {"code", "pre", "samp", "kbd"} and self._code_depth:
            self._code_depth -= 1
        if tag in self._tag_stack:
            while self._tag_stack and self._tag_stack.pop() != tag:
                pass

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        self.visible_parts.append(data)
        if not self._code_depth and data.strip():
            context = self._tag_stack[-1] if self._tag_stack else "body"
            self.prose_parts.append((context, data))


def validate_page(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    inspector = PageInspector()
    inspector.feed(text)

    if not text.startswith("<!DOCTYPE html>"):
        errors.append("missing <!DOCTYPE html>")
    if not text.rstrip().endswith("</html>"):
        errors.append("missing </html>")

    searchable_text = re.sub(r"data:[^\"']+", "", text)
    for match in PLACEHOLDER_RE.findall(searchable_text):
        errors.append(f"template placeholder remains: {match}")
    for needle in MARKERS:
        if needle in searchable_text:
            errors.append(f"template marker remains: {needle}")

    ids = ID_RE.findall(text)
    seen: set[str] = set()
    for id_value in ids:
        if id_value in seen:
            errors.append(f"duplicate id: {id_value}")
        seen.add(id_value)

    id_set = set(ids)
    for anchor in SAME_PAGE_ANCHOR_RE.findall(text):
        if anchor and anchor not in id_set:
            errors.append(f"anchor points to missing id: #{anchor}")

    for ref in LOCAL_ASSET_RE.findall(text):
        if ref.startswith(IGNORE_PREFIXES):
            continue
        clean_ref = ref.split("#", 1)[0].split("?", 1)[0]
        if not clean_ref:
            continue
        target = (path.parent / clean_ref).resolve()
        if clean_ref.endswith("/"):
            target = target / "index.html"
        if not target.exists():
            errors.append(f"broken local reference {ref}")

    # 正文里以文字形式指向的 research/ 路径必须真实存在，且不能指向目录快照。
    # research/ 整个目录不发布，正文指向它读者必然打不开；唯一例外是
    # measured.md——指南规定实测结论以「实测得到」陈述并指向这份登记。
    for mention in RESEARCH_MENTION_RE.findall(text):
        if RESEARCH_ALLOWED_RE.match(mention):
            continue
        target = (path.parent / mention).resolve()
        if not target.exists():
            errors.append(
                f"reference to a missing research file: {mention}"
                " (rewrite as '实测得到' or point to research/measured.md)"
            )
        else:
            errors.append(
                f"reference to unpublished research path: {mention}"
                " (research/ is excluded from the deployed site;"
                " rewrite as '实测得到' or point to research/measured.md)"
            )

    is_wiki_page = "wiki" in path.parts
    if is_wiki_page and path.name == "index.html":
        for name in REQUIRED_WIKI_META:
            if not inspector.meta.get(name):
                errors.append(f"missing metadata: {name}")

        # 主题取值必须是封闭词表内的大类，防止主题再次碎片化
        topics = [
            item.strip()
            for item in re.split(r"[,，]", inspector.meta.get("dojo:topics", ""))
            if item.strip()
        ]
        for topic in topics:
            if topic not in ALLOWED_TOPICS:
                errors.append(
                    f"unknown topic: {topic} (allowed: {', '.join(ALLOWED_TOPICS)})"
                )

        raw_type = inspector.meta.get("dojo:type", "")
        if raw_type and raw_type not in ALLOWED_TYPES:
            errors.append(
                f"unknown type: {raw_type} (allowed: {', '.join(ALLOWED_TYPES)})"
            )
        elif raw_type in TYPE_STYLESHEET:
            expected_css = TYPE_STYLESHEET[raw_type]
            # 部署会给样式链接加 ?v=<sha> 缓存参数，比对时忽略查询串。
            if not any(
                ref.split("?", 1)[0].endswith(f"libs/{expected_css}")
                for ref in inspector.stylesheets
            ):
                errors.append(
                    f"type {raw_type} must reference ../../libs/{expected_css}"
                )

        # 标签取封闭词表内的单一值，防止细粒度标签再次碎片化
        raw_tag = inspector.meta.get("dojo:tag", "")
        if re.search(r"[,，]", raw_tag):
            errors.append(f"dojo:tag must be a single value, got: {raw_tag}")
        elif raw_tag and raw_tag not in ALLOWED_TAGS:
            errors.append(
                f"unknown tag: {raw_tag} (allowed: {', '.join(ALLOWED_TAGS)})"
            )

        description = inspector.meta.get("description", "")
        summary = inspector.meta.get("dojo:summary", "")
        if "$" in description:
            errors.append("description must be plain text; put formulas in dojo:summary")
        if "$$" in summary:
            errors.append("dojo:summary supports inline $...$ formulas only")
        if summary.count("$") % 2:
            errors.append("dojo:summary has unmatched $ delimiter")

    visible_text = " ".join(inspector.visible_parts)
    summary = inspector.meta.get("dojo:summary", "")
    if is_wiki_page and INLINE_MATH_RE.search(f"{summary} {visible_text}"):
        has_katex_js = any("katex" in ref for ref in inspector.scripts)
        has_katex_css = any("katex" in ref for ref in inspector.stylesheets)
        has_auto_render = (
            any("auto-render" in ref for ref in inspector.scripts)
            and "renderMathInElement" in text
        )
        if not has_katex_js or not has_katex_css:
            errors.append("math content requires local KaTeX JS and CSS")
        if not has_auto_render:
            errors.append("math content requires auto-render initialization")

    if is_wiki_page:
        errors.extend(check_bare_math(inspector))
        errors.extend(check_svg_text_math(text))
        errors.extend(check_box_drawing(text))
        if path.name == "index.html":
            errors.extend(check_research_dir(path.parent / "research"))

    return errors


def check_research_dir(research: Path) -> list[str]:
    """research/ 只放符合固定清单的 .md，不混入其他格式与临时文件。

    该目录不随站点发布，但仍要可长期检索：文件名限定为固定集合，审查记录
    按 review-<轮次>.md 连续编号，引文快照放在 sources/ 与 official/ 下。
    """
    errors: list[str] = []
    if not research.is_dir():
        return errors
    for entry in sorted(research.iterdir()):
        if entry.is_dir():
            if entry.name not in RESEARCH_ALLOWED_DIRS:
                errors.append(f"unexpected directory in research/: {entry.name}/")
            continue
        if not entry.name.endswith(".md"):
            errors.append(f"non-markdown file in research/: {entry.name}")
        elif entry.name in RESEARCH_ALLOWED_FILES or REVIEW_FILE_RE.fullmatch(entry.name):
            continue
        else:
            errors.append(f"unexpected file in research/: {entry.name}")
    return errors


def check_bare_math(inspector: PageInspector) -> list[str]:
    """公式定界符之外出现数学 Unicode 字符即视为未渲染公式。

    覆盖标题、summary、正文与列表；代码块内的字符不计入。
    """
    errors: list[str] = []
    for context, chunk in inspector.prose_parts:
        if context in UI_CONTEXT_TAGS:
            continue
        outside = strip_math_segments(chunk)
        hits = sorted(set(BARE_MATH_RE.findall(outside)))
        if hits:
            snippet = " ".join(outside.split())[:70]
            errors.append(
                f"unrendered math characters {''.join(hits)} in <{context}>: {snippet}"
                " (wrap in $...$ so KaTeX renders it)"
            )

    summary = inspector.meta.get("dojo:summary", "")
    hits = sorted(set(BARE_MATH_RE.findall(strip_math_segments(summary))))
    if hits:
        errors.append(
            f"unrendered math characters {''.join(hits)} in dojo:summary"
            " (wrap in $...$ so KaTeX renders it)"
        )
    return errors


def check_svg_text_math(text: str) -> list[str]:
    """SVG 的 <text> 内不得出现公式或数学字符，改用 <foreignObject>。"""
    errors: list[str] = []
    for tag, content in SVG_TEXT_RE.findall(text):
        inner = unescape(re.sub(r"<[^>]+>", "", content))
        if INLINE_MATH_RE.search(inner):
            snippet = " ".join(inner.split())[:60]
            errors.append(
                f"math delimiters inside <{tag}> are not rendered: {snippet}"
                " (use <foreignObject> to hold the formula)"
            )
            continue
        hits = sorted(set(BARE_MATH_RE.findall(inner)))
        if hits:
            snippet = " ".join(inner.split())[:60]
            errors.append(
                f"unrendered math characters {''.join(hits)} inside <{tag}>: {snippet}"
                " (use <foreignObject> with $...$ instead)"
            )
            continue
        for pattern in ASCII_MATH_PATTERNS:
            if pattern.search(inner):
                snippet = " ".join(inner.split())[:60]
                errors.append(
                    f"ascii approximation of math inside <{tag}>: {snippet}"
                    " (use <foreignObject> with $...$ instead)"
                )
                break
    return errors


def check_box_drawing(text: str) -> list[str]:
    """结构图必须用 HTML 或内联 SVG，不使用等宽字符拼出的框线图。"""
    errors: list[str] = []
    for index, block in enumerate(PRE_BLOCK_RE.findall(text), start=1):
        hits = BOX_DRAWING_RE.findall(block)
        if len(hits) >= BOX_DRAWING_MIN_HITS:
            errors.append(
                f"ascii box-drawing diagram in <pre> block #{index}"
                f" ({len(hits)} box characters); use HTML diagram or inline SVG"
            )
    return errors


def check_template(path: Path) -> list[str]:
    """模板必须声明正确类型，且内联样式与共享样式保持一致。

    模板因含占位符不能走 validate_page；但类型写错时，按模板生成的每个
    页面都会错，因此在模板层面单独拦一道。

    模板必须内联样式：它位于 .dojo/templates/<module>/，相对路径
    ../../libs/ 会解析到 .dojo/libs，外链拿不到根目录下的共享样式。
    页面（wiki/<name>/）才外链 libs/dojo-<module>.css。因此模板与共享
    样式是同一份样式的两个落点，必须同步，否则两边会各自漂移。
    """
    errors: list[str] = []
    module = path.parent.name
    expected_css = TYPE_STYLESHEET.get(module)
    if expected_css is None:
        return errors

    text = path.read_text(encoding="utf-8", errors="ignore")
    inspector = PageInspector()
    inspector.feed(text)
    raw_type = inspector.meta.get("dojo:type", "")
    if raw_type != module:
        errors.append(f"template type must be {module}, got: {raw_type or '(missing)'}")

    css_path = Path("libs") / expected_css
    blocks = STYLE_RE.findall(text)
    if not blocks:
        errors.append("template must inline its styles (../../libs resolves to .dojo/libs)")
        return errors
    if not css_path.exists():
        errors.append(f"missing shared stylesheet: {css_path}")
        return errors

    def normalize(css: str) -> str:
        css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
        return re.sub(r"\s+", " ", css).strip()

    if normalize(blocks[0]) != normalize(css_path.read_text(encoding="utf-8")):
        errors.append(
            f"template inline styles drifted from {css_path}; "
            "keep the two copies identical"
        )
    return errors


def main() -> int:
    args = [
        arg for arg in sys.argv[1:] if arg not in {"--templates", "--all"}
    ]
    check_templates = "--templates" in sys.argv

    if check_templates:
        template_root = Path(TEMPLATE_DIR)
        templates = sorted(template_root.glob("*/index.html"))
        failures = [(t, check_template(t)) for t in templates]
        failures = [(t, errs) for t, errs in failures if errs]
        if failures:
            print("template validation failed:")
            for template, errs in failures:
                print(f"- {template}")
                for error in errs:
                    print(f"  - {error}")
            return 1
        print(f"template validation ok: {len(templates)} templates")
        return 0

    if "--all" in sys.argv:
        args = sorted(str(p) for p in Path("wiki").glob("**/*.html"))

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
        print("error: page not found: " + ", ".join(missing))
        return 2

    if len(pages) == 1:
        results = [(pages[0], validate_page(pages[0]))]
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda p: (p, validate_page(p)), pages))

    failed = [(page, errors) for page, errors in results if errors]
    if failed:
        print(f"validation failed: {len(failed)}/{len(results)} pages")
        for page, errors in failed:
            print(f"- {page}")
            for error in errors:
                print(f"  - {error}")
        return 1

    if len(pages) == 1:
        print(f"validation ok: {pages[0]}")
    else:
        print(f"validation ok: {len(pages)} pages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
