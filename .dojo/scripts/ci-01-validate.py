#!/usr/bin/env python3
"""发布闸门：页面、内联脚本、样式、首页目录，一次跑完。

站点部署前必须全绿的全部机械检查，入口只有一个。内容质量不在这里评
（那由独立审查负责）。

    # 页面校验（默认；顺带跑内联脚本语法，需要 node，缺失则跳过）
    python3 .dojo/scripts/ci-01-validate.py <页面...>
    python3 .dojo/scripts/ci-01-validate.py --all

    # 专项
    python3 .dojo/scripts/ci-01-validate.py --templates
    python3 .dojo/scripts/ci-01-validate.py --copy [--fix]
    python3 .dojo/scripts/ci-01-validate.py --js <页面...>
    python3 .dojo/scripts/ci-01-validate.py --css [--module T] [--template P]
                                                 [--confirm] [--fix]
    python3 .dojo/scripts/ci-01-validate.py --catalog --output _site/catalog.json

    # 改共享样式前后的渲染比对
    python3 .dojo/scripts/ci-01-validate.py --diff <before-root> <after-root> <页面...>

各段职责见文件内的分节标题。词表（ALLOWED_TOPICS / ALLOWED_TAGS /
ALLOWED_TYPES）定义在这里，页面校验与目录构建共用同一份。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import functools
import http.server
import json
import posixpath
import re
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit


# ============================================================
# 词表：页面 dojo:topics / dojo:type / dojo:tag 的封闭取值
# 页面校验与首页目录构建共用这一份；新增取值要同步 AGENTS.md
# ============================================================
# 主题封闭词表：页面 dojo:topics 只能从中取值；新增后同步 AGENTS.md。
# 需要新增大类时改这里并同步 AGENTS.md。
ALLOWED_TOPICS = [
    "注意力机制",
    "模型结构",
    "推理系统",
    "内存与缓存",
    "并行与通信",
    "训练与优化",
    "多模态",
    "数学基础",
]

# 页面类型封闭词表：dojo:type 只能从中取值。
ALLOWED_TYPES = [
    "concept",
    "dataflow",
    "note",
    "paper",
]

# 细粒度标签封闭词表：dojo:tag 只取其一，供首页按技术筛选。
# topics 是粗分类，一个页面可属多个。
# tag 回答「这篇讲什么技术」，单一取值。
# 新增取值前先确认它至少有 3 个页面。
ALLOWED_TAGS = [
    "KV cache",
    "MoE",
    "优化器",
    "位置编码",
    "并行与通信",
    "推理加速",
    "推理系统",
    "数学与数值",
    "数据流",
    "模型架构",
    "注意力",
    "网络结构",
    "视觉与多模态",
    "训练",
    "量化",
]
# ============================================================
# 页面校验：结构、元数据、本地引用、数学字符、research 目录、模板
# ============================================================
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
# 每个 dojo:type 对应的共享样式表，防止页面套错模板后仍能通过校验。
# 元组第一项是该类型模板必须内联的样式表；页面引用其中任意一项即可。
# dataflow 有两代：旧页用 cytoscape + dojo-dataflow.css，新页用 ELK + dojo-flow.css。
TYPE_STYLESHEET = {
    "concept": ("dojo-concept.css",),
    "paper": ("dojo-paper.css",),
    "note": ("dojo-note.css",),
    "dataflow": ("dojo-flow.css", "dojo-dataflow.css"),
}
TEMPLATE_DIR = ".dojo/templates"
STYLE_RE = re.compile(r"<style[^>]*>(.*?)</style>", re.S | re.I)
# <noscript> 里的 <style> 是无脚本回退的补充规则（例如让画布控件在无 JS 时
# 显示出来），不是模板的主样式，判断样式来源时要排除。
NOSCRIPT_RE = re.compile(r"<noscript[\s\S]*?</noscript>", re.I)

# 概览页与索引页的入口文案：历史上 A/B 两代并存，模板已固定为 B 世代。
# A 世代 eyebrow「概念快速阅读」/ nav「深度教学 →」/ footer「快速阅读」
# B 世代 eyebrow「概念概览」  / nav「完整说明 →」/ footer「概览」
OVERVIEW_TERM = {"concept": "概览", "paper": "概览"}
DETAIL_TERM = {"concept": "完整说明", "paper": "完整解析"}
TITLE_TAG_RE = re.compile(r"<title>(.*?)</title>", re.S)
EYEBROW_RE = re.compile(r'(<span class="eyebrow">)(.*?)(</span>)', re.S)
NAV_BLOCK_RE = re.compile(r"(<nav>.*?</nav>)", re.S)
FOOTER_BLOCK_RE = re.compile(r"<footer>(.*?)</footer>", re.S)
OVERVIEW_LINK_RE = re.compile(
    r'<a class="overview-link"[^>]*title="[^"]*"[^>]*>.*?</a>', re.S
)

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
    "dataflow.md",
    "concept.md",
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

    parts = path.parts
    is_wiki_page = "wiki" in parts
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
            allowed_css = TYPE_STYLESHEET[raw_type]
            # 部署会给样式链接加 ?v=<sha> 缓存参数，比对时忽略查询串。
            if not any(
                ref.split("?", 1)[0].endswith(f"libs/{css}")
                for css in allowed_css
                for ref in inspector.stylesheets
            ):
                errors.append(
                    f"type {raw_type} must reference one of "
                    + ", ".join(f"../../libs/{css}" for css in allowed_css)
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
        errors.extend(check_body_centering(text))

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


BODY_RULE_RE = re.compile(r"(?:^|[},])\s*body\s*\{([^}]*)\}", re.M | re.S)
BODY_MARGIN_RE = re.compile(r"margin\s*:\s*([^;}]*)")


def check_body_centering(text: str) -> list[str]:
    """页面内联样式覆盖 body 的 margin 时必须保留横向 auto。

    共享样式用 `body { max-width: 1200px; margin: 0 auto }` 把正文居中，
    左侧固定目录的 left 也按正文居中计算。页面若在自己的 <style> 里写
    `body { margin: 0 }`，居中失效、目录会压到标题和正文上；这个覆盖看不到
    报错，只会在 1440–1920px 的视口里表现为文字重叠。
    """
    errors: list[str] = []
    for block in STYLE_RE.findall(text):
        for rule in BODY_RULE_RE.findall(block):
            match = BODY_MARGIN_RE.search(rule)
            if not match:
                continue
            value = match.group(1).strip()
            if "auto" not in value:
                errors.append(
                    f"inline body margin overrides the centering: margin: {value}"
                    " (use 'margin: 0 auto' so the fixed TOC does not overlap)"
                )
    return errors


def check_template(path: Path) -> list[str]:
    """模板必须声明正确类型，且样式只有一份来源。

    模板因含占位符不能走 validate_page；但类型写错时，按模板生成的每个
    页面都会错，因此在模板层面单独拦一道。

    样式的处理有两种，都允许，但都必须保证「只有一份真相」：

      · 内联：散文类模板这样做。模板位于 .dojo/templates/<module>/，
        相对路径 ../../libs/ 会解析到 .dojo/libs，直接打开模板看不到样式；
        内联后模板自身可预览。代价是多一份副本，故必须与共享文件逐字一致。

      · 外链：依赖 JS 的模板这样做。数据流页离开 elk.bundled.js 与
        dojo-flow.js 就渲染不出任何东西，独立预览本就无意义，内联只是
        白白多一份副本。直接引用共享文件，从根上不存在漂移。

    两种都不满足（既没内联也没引用）时报错。
    """
    errors: list[str] = []
    module = path.parent.name
    allowed_css = TYPE_STYLESHEET.get(module)
    if allowed_css is None:
        return errors
    # 模板内联样式时，与元组第一项比对（每类页面只有一个权威样式表）。
    canonical_css = allowed_css[0]

    text = path.read_text(encoding="utf-8", errors="ignore")
    inspector = PageInspector()
    inspector.feed(text)
    raw_type = inspector.meta.get("dojo:type", "")
    if raw_type != module:
        errors.append(f"template type must be {module}, got: {raw_type or '(missing)'}")

    # 只看模板的主样式：先去掉 <noscript> 段，其中的 <style> 属于无脚本回退。
    style_scope = NOSCRIPT_RE.sub("", text)
    blocks = STYLE_RE.findall(style_scope)
    linked = any(
        ref.split("?", 1)[0].endswith(f"libs/{css}")
        for css in allowed_css
        for ref in inspector.stylesheets
    )
    if not blocks and not linked:
        errors.append(
            "template must either inline or link the shared stylesheet: "
            + " or ".join(f"../../libs/{css}" for css in allowed_css)
        )
        return errors

    css_path = Path("libs") / canonical_css
    if not css_path.exists():
        errors.append(f"missing shared stylesheet: {css_path}")
        return errors

    # 外链时没有副本，无从漂移，到此为止。
    if not blocks:
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


# ---------- 概览页文案一致性 ----------


def page_type_of(index_html: str) -> str:
    match = re.search(r'name="dojo:type"\s+content="([^"]+)"', index_html)
    return match.group(1) if match else ""


def fix_overview_copy(text: str, ty: str) -> str:
    """把概览页文案回填到模板的说法。

    只处理 concept / paper 两类。类型未知（拿不到 index.html 的
    dojo:type）时原样返回——否则会拿「概览 / 完整说明」这套说法去改写
    其他类型的页面。
    """
    if ty not in OVERVIEW_TERM:
        return text
    detail = DETAIL_TERM[ty]

    text = text.replace("深度教学页", f"{detail}页").replace("深度教学", detail)
    text = text.replace("快速阅读", "概览")

    def title_sub(match: re.Match) -> str:
        title = match.group(1).strip()
        body = re.sub(r"\s*·\s*[^·]+?\s*·\s*Dojo\s*$", "", title)
        body = re.sub(r"\s*·\s*Dojo\s*$", "", body).strip()
        if body and body != title:
            return f"<title>{body} · 概览 · Dojo</title>"
        return match.group(0)

    text = TITLE_TAG_RE.sub(title_sub, text)

    def eyebrow_sub(match: re.Match) -> str:
        inner = match.group(2)
        tail = inner.split("·", 1)[1].strip() if "·" in inner else ""
        label = "概念概览" if ty == "concept" else "论文概览"
        body = f"{label} · {tail}" if tail else label
        return f"{match.group(1)}{body}{match.group(3)}"

    text = EYEBROW_RE.sub(eyebrow_sub, text)

    def nav_sub(match: re.Match) -> str:
        block = match.group(1)
        block = block.replace(r"$\leftarrow$", "←").replace(r"$\to$", "→")
        return re.sub(
            r'(<a href="index\.html"[^>]*>).*?(</a>)',
            lambda m: m.group(1) + detail + " →" + m.group(2),
            block,
            flags=re.S,
        )

    text = NAV_BLOCK_RE.sub(nav_sub, text)
    footer_text = f"Dojo · 概览 · 机制细节见{detail}页"
    return FOOTER_BLOCK_RE.sub(lambda m: f"<footer>{footer_text}</footer>", text)


def fix_index_link(text: str) -> str:
    if '<a class="overview-link"' not in text:
        return text
    return OVERVIEW_LINK_RE.sub(
        '<a class="overview-link" href="overview.html" title="查看概览">概览</a>',
        text,
    )


def convert_copy(path: Path, fix: bool) -> list[str]:
    changes: list[str] = []
    if path.name == "overview.html":
        index = path.parent / "index.html"
        ty = page_type_of(index.read_text(encoding="utf-8")) if index.exists() else ""
        original = path.read_text(encoding="utf-8")
        updated = fix_overview_copy(original, ty)
        if updated != original:
            changes.append(f"{path}: 概览页文案")
            if fix:
                path.write_text(updated, encoding="utf-8")
    elif path.name == "index.html":
        original = path.read_text(encoding="utf-8")
        updated = fix_index_link(original)
        if updated != original:
            changes.append(f"{path}: 概览入口链接")
            if fix:
                path.write_text(updated, encoding="utf-8")
    return changes


# ---------- 内联脚本语法 ----------

# HTML 规范里浏览器会当作 JavaScript 执行的 type 取值
JS_MIME_TYPES = frozenset(
    {
        "application/ecmascript",
        "application/javascript",
        "application/x-ecmascript",
        "application/x-javascript",
        "text/ecmascript",
        "text/javascript",
        "text/javascript1.0",
        "text/javascript1.1",
        "text/javascript1.2",
        "text/javascript1.3",
        "text/javascript1.4",
        "text/javascript1.5",
        "text/jscript",
        "text/livescript",
        "text/x-ecmascript",
        "text/x-javascript",
        "module",
    }
)

# 老式 "<!-- ... //-->" 包裹，去掉注释标记后再交给 node
LEGACY_OPEN_RE = re.compile(r"^\s*<!--")
LEGACY_CLOSE_RE = re.compile(r"//-->\s*$")


class ScriptCollector(HTMLParser):
    """收集页面里所有内联脚本的源码。

    用 HTMLParser 而不是正则：它按 HTML 规则解析属性（引号内的 > 不会截断
    标签）、正确跳过注释里的示例代码，也不要求 </script> 后面直接跟 >。
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.blocks: list[str] = []
        self._attrs: dict[str, str] = {}
        self._buffer: list[str] = []
        self._inside = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "script":
            return
        self._attrs = {name.lower(): (value or "") for name, value in attrs}
        self._buffer = []
        self._inside = True

    def handle_data(self, data: str) -> None:
        if self._inside:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "script" or not self._inside:
            return
        self._inside = False
        if "src" in self._attrs:
            return
        kind = self._attrs.get("type", "").split(";")[0].strip().lower()
        if kind and kind not in JS_MIME_TYPES:
            return
        body = LEGACY_OPEN_RE.sub("", "".join(self._buffer))
        body = LEGACY_CLOSE_RE.sub("", body)
        if body.strip():
            self.blocks.append(body)


def check_inline_js(pages: list[Path]) -> tuple[list[str], int, bool]:
    """内联 <script> 的语法检查。返回 (错误, 块数, 是否执行)。

    只查语法，不执行代码：拼错括号或引号会在这里拦下，但运行时报错
    （加载顺序、未定义全局、第三方包内部抛错）查不出来。

    逐个块起 node 进程时，200 多个块要起 200 多次进程，耗时以秒计。
    这里在单个 node 进程内用 vm.Script 逐个编译，语义与 --check 一致，
    失败信息按 FILE: 前缀分段，便于映射回页面与块号。

    node 不存在时返回「未执行」，由调用方决定是否提示，不阻断其他检查。
    """
    if shutil.which("node") is None:
        return [], 0, False

    errors: list[str] = []
    total = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        targets: list[tuple[str, Path]] = []
        for page in pages:
            parser = ScriptCollector()
            parser.feed(page.read_text(encoding="utf-8", errors="replace"))
            for index, body in enumerate(parser.blocks, start=1):
                total += 1
                target = tmp / f"{page.parent.name}-{page.stem}-{index}.js"
                target.write_text(body, encoding="utf-8")
                targets.append((f"{page} 内联 <script> #{index}", target))

        if not targets:
            return errors, total, True

        script = (
            "const fs=require('fs'),vm=require('vm');"
            "let failed=false;"
            "for(const f of process.argv.slice(1)){"
            "try{new vm.Script(fs.readFileSync(f,'utf8'),{filename:f});}"
            "catch(e){failed=true;process.stderr.write('FILE:'+f+'\\n'"
            "+e.message.replace(/\\s+/g,' ')+'\\n');}"
            "}"
            "process.exit(failed?1:0);"
        )
        proc = subprocess.run(
            ["node", "-e", script, *[str(t) for _, t in targets]],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).replace(str(tmp.resolve()) + "/", "")
            blocks: dict[str, str] = {}
            current = None
            for line in detail.splitlines():
                if line.startswith("FILE:"):
                    current = line[len("FILE:"):].strip()
                    blocks[current] = ""
                elif current:
                    blocks[current] += (" " if blocks[current] else "") + line.strip()
            for label, target in targets:
                message = blocks.get(str(target))
                if message:
                    errors.append(f"{label}: {' '.join(message.split())}")
    return errors, total, True


def check_overview_copy(fix: bool) -> int:
    targets = sorted(Path("wiki").glob("*/overview.html")) + sorted(
        Path("wiki").glob("*/index.html")
    )
    changes: list[str] = []
    for path in targets:
        changes.extend(convert_copy(path, fix))

    if not changes:
        print(f"no copy drift in {len(targets)} files")
        return 0
    if fix:
        print(f"已统一 {len(changes)} 处：")
        for line in changes:
            print(f"  {line}")
        return 0
    print(f"validation failed: {len(changes)} 处文案不一致")
    for line in changes:
        print(f"- {line}")
    return 1


# ============================================================
# 样式：死规则检测，以及改样式前后的渲染比对
# ============================================================
SCRIPT_RE = re.compile(r"<script[^>]*>(.*?)</script>", re.S | re.I)
CLASS_ATTR_RE = re.compile(r"""\bclass\s*=\s*["']([^"']*)["']""", re.I)
IDENT_RE = re.compile(r"[A-Za-z_][\w-]*")
CLASS_TOKEN_RE = re.compile(r"\.(-?[A-Za-z_][\w-]*)")
COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)

# 库在运行时注入的 class，静态标记里查不到，必须视为已用
RUNTIME_PREFIXES = ("katex", "token", "language-", "prism", "line-numbers")

# --confirm 用的无头浏览器
CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
)
CLASS_PROBE = """<script>
window.addEventListener('load', function () {
  var seen = {};
  var nodes = document.querySelectorAll('*');
  for (var i = 0; i < nodes.length; i++) {
    var list = nodes[i].classList;
    for (var j = 0; j < list.length; j++) seen[list[j]] = 1;
  }
  document.title = 'CLASSES:' + JSON.stringify(Object.keys(seen));
});
</script>
"""
CLASS_TITLE_RE = re.compile(r"<title>CLASSES:(.*?)</title>", re.S)


def harvest(document: str) -> set[str]:
    """从一份文档里收集可能被应用到的 class 名。"""
    names: set[str] = set()
    for value in CLASS_ATTR_RE.findall(document):
        names.update(value.split())
    for script in SCRIPT_RE.findall(document):
        names.update(IDENT_RE.findall(script))
    return names


def build_usage(targets: list[Path], extra: list[Path]) -> set[str]:
    names: set[str] = set()
    for path in list(targets) + list(extra):
        names |= harvest(path.read_text(encoding="utf-8"))
    return names


def iter_rules(css: str):
    """遍历 CSS 规则，产出 (选择器, 起始偏移, 结束偏移)。"""
    text = COMMENT_RE.sub(lambda m: " " * len(m.group(0)), css)
    stack: list[tuple[str, int]] = []
    start = 0
    cursor = 0
    while cursor < len(text):
        char = text[cursor]
        if char == "{":
            stack.append((text[start:cursor].strip(), cursor))
        elif char == "}":
            if stack:
                selector, _ = stack.pop()
                if selector and not selector.startswith("@"):
                    yield selector, start, cursor + 1
            start = cursor + 1
        elif char == ";" and not stack:
            start = cursor + 1
        cursor += 1


def selector_is_dead(selector: str, used: set[str]) -> bool:
    """判断不了的一律返回 False（保留）。"""
    if "[" in selector or ":not(" in selector or "(" in selector or "::" in selector:
        return False
    tokens = CLASS_TOKEN_RE.findall(selector)
    if not tokens:
        return False
    return any(
        token not in used
        and not token.startswith(RUNTIME_PREFIXES)
        for token in tokens
    )


def split_selectors(selector: str) -> list[str]:
    return [part.strip() for part in selector.split(",") if part.strip()]


def process(path: Path, usage: set[str], fix: bool) -> list[str]:
    document = path.read_text(encoding="utf-8")
    removed: list[str] = []
    for block in reversed(list(STYLE_RE.finditer(document))):
        css = block.group(1)
        spans: list[tuple[int, int, str]] = []
        for selector, local_start, local_end in iter_rules(css):
            parts = split_selectors(selector)
            if parts and all(selector_is_dead(part, usage) for part in parts):
                spans.append((local_start, local_end, selector.strip()))
        if not spans:
            continue
        removed.extend(s for _, _, s in spans)
        if fix:
            new_css = css
            for local_start, local_end, _ in reversed(spans):
                new_css = new_css[:local_start] + new_css[local_end:]
            new_css = re.sub(r"\n{3,}", "\n\n", new_css).rstrip() + "\n"
            document = document[: block.start(1)] + new_css + document[block.end(1) :]
    if fix and removed:
        path.write_text(document, encoding="utf-8")
    return removed


def pages_of_type(root: Path, kind: str) -> list[Path]:
    """wiki/ 下所有 dojo:type 等于 kind 的页面。"""
    pattern = re.compile(r'name="dojo:type"\s+content="' + re.escape(kind) + r'"')
    return [
        path
        for path in sorted(root.glob("wiki/*/index.html"))
        if pattern.search(path.read_text(encoding="utf-8"))
    ]


def find_chrome() -> str | None:
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def live_classes(chrome: str, page: Path) -> set[str]:
    """把页面跑起来，返回脚本执行完之后 DOM 上真实存在的 class 名。"""
    document = page.read_text(encoding="utf-8", errors="replace")
    if "<head>" in document:
        document = document.replace("<head>", "<head>" + CLASS_PROBE, 1)
    else:
        document = CLASS_PROBE + document
    with tempfile.TemporaryDirectory() as tmp:
        probe_page = Path(tmp) / page.name
        probe_page.write_text(document, encoding="utf-8")
        proc = subprocess.run(
            [
                chrome,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--virtual-time-budget=4000",
                "--dump-dom",
                probe_page.as_uri(),
            ],
            capture_output=True,
            text=True,
            timeout=90,
        )
    match = CLASS_TITLE_RE.search(proc.stdout)
    if not match:
        raise RuntimeError(f"probe did not report on {page}")
    return set(json.loads(match.group(1)))


def confirm_removals(candidates: set[str], pages: list[Path]) -> int:
    """在真实浏览器里复核候选 class 确实不再出现。

    返回 0 表示全部确认可删；1 表示仍有候选在用（不能删）；2 表示环境错误。
    """
    chrome = find_chrome()
    if chrome is None:
        print("error: --confirm 需要 Chrome/Chromium，未找到", file=sys.stderr)
        return 2

    live: set[str] = set()
    for page in pages:
        try:
            live |= live_classes(chrome, page)
        except Exception as error:  # noqa: BLE001 - 报告后返回
            print(f"error: {error}", file=sys.stderr)
            return 2

    survivors = sorted(c for c in candidates if c in live)
    if survivors:
        print(f"{len(survivors)} class 在浏览器中仍存在，不能删：")
        for name in survivors:
            print(f"  ! .{name}")
        return 1
    print(f"复核通过：{len(candidates)} 个候选 class 在 {len(pages)} 个页面上均未出现")
    return 0


# ---------- 改样式前后的渲染比对 ----------

# 逐元素取这些计算样式与几何，两侧不同即报出。
DIFF_PROPS = (
    "display,position,top,right,bottom,left,float,clear,zIndex,"
    "width,height,minWidth,minHeight,maxWidth,maxHeight,"
    "marginTop,marginRight,marginBottom,marginLeft,"
    "paddingTop,paddingRight,paddingBottom,paddingLeft,"
    "borderTopWidth,borderRightWidth,borderBottomWidth,borderLeftWidth,"
    "borderTopStyle,borderRightStyle,borderBottomStyle,borderLeftStyle,"
    "borderTopColor,borderRightColor,borderBottomColor,borderLeftColor,"
    "borderTopLeftRadius,borderTopRightRadius,borderBottomLeftRadius,borderBottomRightRadius,"
    "background,backgroundColor,backgroundImage,backgroundSize,backgroundPosition,"
    "color,opacity,visibility,overflow,overflowX,overflowY,"
    "fontFamily,fontSize,fontWeight,fontStyle,lineHeight,letterSpacing,"
    "textAlign,textDecorationLine,textTransform,whiteSpace,wordBreak,textIndent,"
    "verticalAlign,listStyleType,listStylePosition,"
    "flexDirection,flexWrap,justifyContent,alignItems,alignSelf,flexGrow,flexShrink,flexBasis,gap,"
    "gridTemplateColumns,gridTemplateRows,gridColumn,gridRow,"
    "transform,transformOrigin,boxShadow,boxSizing,cursor"
).split(",")

DIFF_PROBE = """
<script>
window.addEventListener('load', function () {
  // 等一拍再测：KaTeX 的 auto-render 在 defer 脚本的 onload 里跑，与 window
  // load 几乎同时，立刻取字形度量会抖动。用 setTimeout 而不是 document.fonts
  // .ready——后者在 --virtual-time-budget 下不兑现，探针会永远不执行。
  setTimeout(function () {
    var PROPS = %s;
    function pathOf(el) {
      var parts = [];
      var node = el;
      while (node && node.nodeType === 1) {
        var index = 0, sib = node;
        while ((sib = sib.previousElementSibling)) index++;
        parts.unshift(node.tagName + '[' + index + ']');
        node = node.parentElement;
      }
      return parts.join('/');
    }
    var SKIP = {HEAD:1, STYLE:1, LINK:1, SCRIPT:1, META:1, TITLE:1, BASE:1};
    var rows = [];
    var nodes = document.querySelectorAll('*');
    for (var i = 0; i < nodes.length; i++) {
      var el = nodes[i];
      if (SKIP[el.tagName]) continue;
      var cs = getComputedStyle(el);
      var style = '';
      for (var j = 0; j < PROPS.length; j++) style += PROPS[j] + ':' + cs[PROPS[j]] + ';';
      var r = el.getBoundingClientRect();
      var box = [r.x, r.y, r.width, r.height].map(function (v) {
        return Math.round(v * 100) / 100;
      }).join(',');
      rows.push(pathOf(el) + '|' + box + '|' + style);
    }
    var payload = JSON.stringify({count: nodes.length, rows: rows});
    var h = 5381;
    for (var k = 0; k < payload.length; k++) h = ((h * 33) ^ payload.charCodeAt(k)) >>> 0;
    document.title = 'RD:' + nodes.length + ':' + h;
    document.documentElement.setAttribute('data-rd-rows', payload);
  }, 400);
});
</script>
"""

DIFF_TITLE_RE = re.compile(r"<title>RD:(\d+):(\d+)</title>")
DIFF_ROWS_RE = re.compile(r'data-rd-rows="(.*?)"\s*>', re.S)


def make_probe_handler(probe: str):
    """在响应的 HTML 里注入探针。走 HTTP 而不是 file://——file:// 下外链 CSS
    不会被加载，验证共享样式时会得到假差异。"""

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - 基类命名
            path = self.translate_path(self.path)
            target = Path(path)
            if target.is_dir():
                target = target / "index.html"
            if target.suffix == ".html" and target.exists():
                body = target.read_text(encoding="utf-8", errors="replace")
                body = (
                    body.replace("<head>", "<head>" + probe, 1)
                    if "<head>" in body else probe + body
                )
                payload = body.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            super().do_GET()

        def log_message(self, *args):  # 静默
            pass

    return Handler


def serve_tree(root: Path):
    handler = functools.partial(
        make_probe_handler(DIFF_PROBE % json.dumps(DIFF_PROPS)), directory=str(root)
    )
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def snapshot(chrome: str, url: str, want_rows: bool = False):
    """返回 (元素数, 计算样式指纹)；want_rows 时额外返回逐元素明细。"""
    try:
        proc = subprocess.run(
            [chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
             "--virtual-time-budget=8000", "--dump-dom", url],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        return None
    match = DIFF_TITLE_RE.search(proc.stdout)
    if not match:
        return None
    digest = (int(match.group(1)), int(match.group(2)))
    if not want_rows:
        return digest
    rows_match = DIFF_ROWS_RE.search(proc.stdout)
    rows = json.loads(unescape(rows_match.group(1)))["rows"] if rows_match else []
    return digest + (rows,)


def explain_diff(before_rows, after_rows, limit=5) -> list[str]:
    """逐元素找出第一处差异，并指出是哪个 CSS 属性变了。"""
    lines = []
    for index, (left, right) in enumerate(zip(before_rows, after_rows)):
        if left == right:
            continue
        lpath, lbox, lstyle = left.split("|", 2)
        rpath, rbox, rstyle = right.split("|", 2)
        lines.append(f"  element #{index}")
        if lpath != rpath:
            lines.append(f"    path  {lpath} -> {rpath}")
        if lbox != rbox:
            lines.append(f"    box   {lbox} -> {rbox}")
        lprops = dict(kv.split(":", 1) for kv in lstyle.split(";") if ":" in kv)
        rprops = dict(kv.split(":", 1) for kv in rstyle.split(";") if ":" in kv)
        for key in lprops:
            if lprops.get(key) != rprops.get(key):
                lines.append(f"    {key}: {lprops.get(key)!r} -> {rprops.get(key)!r}")
        if len(lines) >= limit * 8:
            break
    if len(before_rows) != len(after_rows):
        lines.append(f"  element count {len(before_rows)} -> {len(after_rows)}")
    return lines


def run_render_diff(args: list[str]) -> int:
    """--diff 子命令：<before-root> <after-root> <页面...> [--detail] [--jobs N]"""
    detail = "--detail" in args
    jobs = 1
    if "--jobs" in args:
        index = args.index("--jobs")
        jobs = max(1, int(args[index + 1]))
        del args[index : index + 2]
    args = [a for a in args if a != "--detail"]
    if len(args) < 3:
        print("用法：--diff <before-root> <after-root> <相对路径...> [--detail] [--jobs N]")
        return 2
    before_root, after_root = Path(args[0]), Path(args[1])
    pages = args[2:]

    chrome = find_chrome()
    if chrome is None:
        print("error: no Chrome/Chromium found", file=sys.stderr)
        return 2

    before_server, before_port = serve_tree(before_root)
    after_server, after_port = serve_tree(after_root)

    def check(rel: str):
        before_file = before_root / rel
        after_file = after_root / rel
        if not before_file.exists() or not after_file.exists():
            return f"ERR  {rel}  (missing on one side)"
        before = snapshot(chrome, f"http://127.0.0.1:{before_port}/{rel}", detail)
        after = snapshot(chrome, f"http://127.0.0.1:{after_port}/{rel}", detail)
        if before is None or after is None:
            return f"ERR  {rel}  (page did not report; check the probe)"
        if before[:2] == after[:2]:
            return f"ok   {rel}  ({before[0]} elements)"
        lines = [f"DIFF {rel}  before={before[:2]} after={after[:2]}"]
        if detail and len(before) > 2 and len(after) > 2:
            lines.extend(explain_diff(before[2], after[2]))
        return "\n".join(lines)

    failed = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            for line in pool.map(check, pages):
                print(line, flush=True)
                if line.startswith(("DIFF", "ERR")):
                    failed += 1
    finally:
        before_server.shutdown()
        after_server.shutdown()

    print(f"\n{len(pages) - failed}/{len(pages)} pages identical", flush=True)
    return 1 if failed else 0


# ============================================================
# 首页目录：扫描 wiki 生成 catalog.json（词表在文件顶部）
# ============================================================
SPACE_RE = re.compile(r"\s+")
IGNORED_SCHEMES = {"http", "https", "mailto", "javascript", "data", "tel"}


def clean_text(value: str) -> str:
    return SPACE_RE.sub(" ", value).strip()


def split_topics(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[,，]", value) if item.strip()]



class WikiHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.hrefs: list[str] = []
        self.title_parts: list[str] = []
        self.h1_parts: list[str] = []
        self.paragraphs: list[str] = []
        self.leads: list[str] = []
        self._title_depth = 0
        self._h1_depth = 0
        self._paragraph_depth = 0
        self._ignored_depth = 0
        self._paragraph_parts: list[str] = []
        self._paragraph_is_lead = False

    def handle_starttag(self, tag: str, attrs: list[tuple]) -> None:
        values = dict(attrs)
        if tag in {"script", "style"}:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag == "meta":
            name = values.get("name", "").strip().lower()
            content = values.get("content", "").strip()
            if name and content:
                self.meta[name] = content
        elif tag == "a":
            href = values.get("href")
            if href:
                self.hrefs.append(href.strip())
        elif tag == "title":
            self._title_depth += 1
        elif tag == "h1":
            self._h1_depth += 1
        elif tag == "p":
            self._paragraph_depth += 1
            self._paragraph_parts = []
            classes = values.get("class", "").split()
            self._paragraph_is_lead = bool({"lead", "summary"} & set(classes))

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1
            return
        if self._ignored_depth:
            return
        if tag == "title" and self._title_depth:
            self._title_depth -= 1
        elif tag == "h1" and self._h1_depth:
            self._h1_depth -= 1
        elif tag == "p" and self._paragraph_depth:
            paragraph = clean_text(" ".join(self._paragraph_parts))
            if paragraph:
                self.paragraphs.append(paragraph)
                if self._paragraph_is_lead:
                    self.leads.append(paragraph)
            self._paragraph_depth -= 1
            self._paragraph_parts = []
            self._paragraph_is_lead = False

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if self._title_depth:
            self.title_parts.append(data)
        if self._h1_depth:
            self.h1_parts.append(data)
        if self._paragraph_depth:
            self._paragraph_parts.append(data)


def discover_pages(root: Path) -> list[Path]:
    return sorted((root / "wiki").glob("*/index.html"))


def git_first_seen_dates(root: Path) -> dict[str, str]:
    """Return {posix path: YYYY-MM-DD} for each wiki file's first commit.

    Uses a single `git log --reverse` pass; a file's date is the earliest
    commit in which the path appears (added or renamed into place). Falls
    back to an empty dict when git history is unavailable (e.g. shallow).
    """
    try:
        result = subprocess.run(
            [
                "git",
                "log",
                "--reverse",
                "--format=%aI",
                "--name-only",
                "--",
                "wiki/",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {}

    dates: dict[str, str] = {}
    current: Optional[str] = None
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}T.*", stripped):
            current = stripped[:10]
        elif stripped.startswith("wiki/") and current:
            dates.setdefault(stripped, current)
    return dates


def parse_page(root: Path, page_path: Path) -> dict:
    relative = page_path.relative_to(root).as_posix()
    parser = WikiHTMLParser()
    parser.feed(page_path.read_text(encoding="utf-8", errors="ignore"))
    h1 = clean_text(" ".join(parser.h1_parts))
    title_tag = clean_text(" ".join(parser.title_parts))
    title = h1 or title_tag or page_path.parent.name
    description = (
        parser.meta.get("description")
        or (parser.leads[0] if parser.leads else "")
        or (parser.paragraphs[0] if parser.paragraphs else "")
    )
    summary_meta = parser.meta.get("dojo:summary", "").strip()
    summary = summary_meta or description
    page_topics = split_topics(parser.meta.get("dojo:topics", ""))
    invalid_topics = [t for t in page_topics if t not in ALLOWED_TOPICS]

    return {
        "id": relative,
        "path": relative,
        "title": title,
        "description": clean_text(description),
        "summary": clean_text(summary),
        "type": parser.meta.get("dojo:type", "").strip() or "unknown",
        "topics": page_topics,
        "tag": parser.meta.get("dojo:tag", "").strip(),
        "_has_summary": bool(summary_meta),
        "_invalid_topics": invalid_topics,
        "_hrefs": parser.hrefs,
    }


def normalize_target(source: str, href: str) -> Optional[str]:
    parts = urlsplit(href)
    if parts.scheme.lower() in IGNORED_SCHEMES or parts.netloc or not parts.path:
        return None
    base = posixpath.dirname(source)
    target = posixpath.normpath(posixpath.join(base, parts.path))
    if parts.path.endswith("/"):
        target = posixpath.join(target, "index.html")
    if target.endswith("/overview.html"):
        target = target[: -len("overview.html")] + "index.html"
    if not target.startswith("wiki/") or not target.endswith("/index.html"):
        return None
    if target == source:
        return None
    return target


def build_catalog(root: Path) -> dict:
    root = root.resolve()
    first_seen = git_first_seen_dates(root)
    pages = [parse_page(root, path) for path in discover_pages(root)]
    for page in pages:
        page["date"] = first_seen.get(page["path"], "")
    page_ids = {page["id"] for page in pages}
    edge_counts: Counter = Counter()
    warnings: list[dict] = []

    for page in pages:
        if not page.pop("_has_summary"):
            warnings.append({"type": "missing_summary", "source": page["id"]})
        if page["type"] == "unknown":
            warnings.append({"type": "unclassified", "source": page["id"]})
        if not page["topics"]:
            warnings.append({"type": "missing_topics", "source": page["id"]})
        for topic in page.pop("_invalid_topics"):
            warnings.append({"type": "unknown_topic", "source": page["id"], "topic": topic})
        if not page["tag"]:
            warnings.append({"type": "missing_tag", "source": page["id"]})
        for href in page.pop("_hrefs"):
            target = normalize_target(page["id"], href)
            if target is None:
                continue
            if target not in page_ids:
                warnings.append(
                    {"type": "missing_target", "source": page["id"], "target": target}
                )
                continue
            edge_counts[(page["id"], target)] += 1

    incoming: dict[str, list[str]] = defaultdict(list)
    outgoing: dict[str, list[str]] = defaultdict(list)
    edges: list[dict] = []
    for (source, target), count in sorted(edge_counts.items()):
        incoming[target].append(source)
        outgoing[source].append(target)
        edges.append(
            {
                "id": f"{source}::{target}",
                "source": source,
                "target": target,
                "count": count,
            }
        )

    for page in pages:
        page["incoming"] = sorted(incoming[page["id"]])
        page["outgoing"] = sorted(outgoing[page["id"]])
        page["incoming_count"] = len(page["incoming"])
        page["outgoing_count"] = len(page["outgoing"])

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pages": sorted(pages, key=lambda page: (page["title"].casefold(), page["id"])),
        "edges": edges,
        "warnings": sorted(
            warnings,
            key=lambda item: (
                item.get("type", ""),
                item.get("source", ""),
                item.get("target", ""),
            ),
        ),
    }


def write_catalog(catalog: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# ============================================================
# 入口调度
# ============================================================


def run_css_check(argv: list[str]) -> int:
    """--css：对每个页面类型跑死规则检测。

    默认覆盖全部四类（concept / paper / note / dataflow）。用 --module 指定
    单个类型、--template 指定模板路径；--confirm 在浏览器里复核候选；
    --fix 直接删。
    """
    fix = "--fix" in argv
    confirm = "--confirm" in argv
    module = None
    template = None
    if "--module" in argv:
        i = argv.index("--module")
        module = argv[i + 1]
    if "--template" in argv:
        i = argv.index("--template")
        template = Path(argv[i + 1])

    modules = [module] if module else list(ALLOWED_TYPES)
    total_removed = 0
    for name in modules:
        if name not in ALLOWED_TYPES:
            print(
                f"error: 未知类型 {name!r}（可选：{', '.join(ALLOWED_TYPES)}）",
                file=sys.stderr,
            )
            return 2
        tpl = template or Path(f".dojo/templates/{name}/index.html")
        # 模板是这一类页面样式的权威副本，缺了就无从检查；静默跳过会让
        # 模板被误删时无人发觉，所以这里直接报错。
        if not tpl.exists():
            print(f"error: 模板不存在：{tpl}", file=sys.stderr)
            return 2
        pages = pages_of_type(Path("."), name)
        if not pages:
            print(f"error: no pages with dojo:type={name}", file=sys.stderr)
            return 2
        targets = [tpl] + pages
        usage = build_usage(targets, pages)
        candidates: set[str] = set()
        removed_here = 0
        for path in targets:
            removed = process(path, usage, fix)
            removed_here += len(removed)
            candidates.update(CLASS_TOKEN_RE.findall("\n".join(removed)))
            if removed:
                print(f"{path}: {len(removed)} rules")
                for selector in removed:
                    print(f"  - {selector}")
        if confirm and candidates:
            # 复核在真实页面里做：模板含占位符、相对路径也解析不到，
            # 浏览器跑模板没有意义。
            status = confirm_removals(candidates, pages)
            if status != 0:
                return status
        total_removed += removed_here
        if not removed_here:
            print(f"no dead css in {len(targets)} files")

    if total_removed:
        print(f"\n{'cleaned' if fix else 'found'} {total_removed} dead rules")
        return 0 if fix else 1
    return 0


def run_catalog(argv: list[str]) -> int:
    """--catalog：构建首页用的 catalog.json。"""
    parser = argparse.ArgumentParser(prog="ci-01-validate --catalog")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args([a for a in argv if a != "--catalog"])

    catalog = build_catalog(args.root)
    write_catalog(catalog, args.output)
    print(
        f"generated {len(catalog['pages'])} pages, "
        f"{len(catalog['edges'])} edges, "
        f"{len(catalog['warnings'])} warnings -> {args.output}"
    )
    return 0


def main() -> int:
    argv = sys.argv[1:]

    if "--diff" in argv:
        i = argv.index("--diff")
        return run_render_diff(argv[i + 1:])
    if "--catalog" in argv:
        return run_catalog(argv)
    if "--css" in argv:
        return run_css_check(argv)
    if "--copy" in argv:
        return check_overview_copy("--fix" in argv)

    if "--templates" in argv:
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

    args = [a for a in argv if a not in {"--all", "--js"}]
    js_only = "--js" in argv
    if "--all" in argv:
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

    if js_only:
        js_errors, total, ran = check_inline_js(pages)
        if not ran:
            print("error: node not found in PATH; install Node.js to run this check")
            return 2
        if js_errors:
            print(f"inline script check failed: {len(js_errors)} of {total} blocks")
            for error in js_errors:
                print(f"- {error}")
            return 1
        print(f"inline script check ok: {len(pages)} pages, {total} blocks")
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
        print(f"validation failed: {len(failed)}/{len(results)} pages")
        for page, errors in failed:
            print(f"- {page}")
            for error in errors:
                print(f"  - {error}")
        return 1

    suffix = f"，内联脚本 {js_total} 块" if js_ran else ""
    if len(pages) == 1:
        print(f"validation ok: {pages[0]}{suffix}")
    else:
        print(f"validation ok: {len(pages)} pages{suffix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
