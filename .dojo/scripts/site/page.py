"""页面与模板的校验：HTML 结构、元数据、本地引用、数学字符、research/ 目录、模板，以及内联脚本语法。

由 validate.py 调度，也可以被 archive/legacy-validate.py 改写表后复用。
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vocab import ALLOWED_STATUS, ALLOWED_TAGS, ALLOWED_TOPICS, ALLOWED_TYPES  # noqa: E402


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

MARKERS = ("@content", "TODO", "TBD")
REQUIRED_WIKI_META = (
    "description",
    "dojo:summary",
    "dojo:type",
    "dojo:topics",
    "dojo:tag",
)
# 每个 dojo:type 对应的共享样式表，防止页面套错模板后仍能通过校验。
# 元组第一项是该类型模板必须内联的样式表；页面引用其中任意一项即可。
# 这里只放概念页。已暂停的论文、学习记录、数据流三类，它们的表在
# archive/legacy-validate.py，那里补回表并打开对 outdated 页面的校验。
TYPE_STYLESHEET = {
    "concept": ("dojo-concept.css",),
}
# 页面标了 dojo:status=outdated 就整页跳过，不校验：旧概念页和已暂停类别的页面都是。
# archive/legacy-validate.py 把它关掉。
SKIP_OUTDATED = True
OUTDATED_RE = re.compile(r'name="dojo:status"\s+content="outdated"')
TEMPLATE_DIR = ".dojo/templates"
STYLE_RE = re.compile(r"<style[^>]*>(.*?)</style>", re.S | re.I)
# <noscript> 里的 <style> 是无脚本回退的补充规则（例如让画布控件在无 JS 时
# 显示出来），不是模板的主样式，判断样式来源时要排除。
NOSCRIPT_RE = re.compile(r"<noscript[\s\S]*?</noscript>", re.I)

# 概览页与索引页的入口文案：历史上 A/B 两代并存，模板已固定为 B 世代。
# A 世代 eyebrow「概念快速阅读」/ nav「深度教学 →」/ footer「快速阅读」
# B 世代 eyebrow「概念概览」  / nav「完整说明 →」/ footer「概览」

# research/ 允许的文件名集合。目录只放 .md，审查轮次按固定名字编号，sources/ 存放资料。
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
    "content.md",
    "conflicts.md",
}
RESEARCH_ALLOWED_DIRS = {"sources"}
REVIEW_FILE_RE = re.compile(r"(?:outline|content)-review-\d+\.md")
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


def is_outdated(path: Path) -> bool:
    """wiki 页面（含同目录的 overview.html）的 index.html 是否标了 dojo:status=outdated。"""
    if not SKIP_OUTDATED or "wiki" not in path.parts:
        return False
    index = path if path.name == "index.html" else path.parent / "index.html"
    if not index.exists():
        return False
    return bool(OUTDATED_RE.search(index.read_text(encoding="utf-8", errors="ignore")))


def validate_page(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    inspector = PageInspector()
    inspector.feed(text)

    if not text.startswith("<!DOCTYPE html>"):
        errors.append("缺少 <!DOCTYPE html>")
    if not text.rstrip().endswith("</html>"):
        errors.append("缺少 </html>")

    searchable_text = re.sub(r"data:[^\"']+", "", text)
    for match in PLACEHOLDER_RE.findall(searchable_text):
        errors.append(f"模板占位符没替换：{match}")
    for needle in MARKERS:
        if needle in searchable_text:
            errors.append(f"模板标记没清掉：{needle}")

    ids = ID_RE.findall(text)
    seen: set[str] = set()
    for id_value in ids:
        if id_value in seen:
            errors.append(f"id 重复：{id_value}")
        seen.add(id_value)

    id_set = set(ids)
    for anchor in SAME_PAGE_ANCHOR_RE.findall(text):
        if anchor and anchor not in id_set:
            errors.append(f"锚点指向不存在的 id：#{anchor}")

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
            errors.append(f"本地引用不存在：{ref}")

    # 正文里以文字形式指向的 research/ 路径必须真实存在，且不能指向目录快照。
    # research/ 整个目录不发布，正文指向它读者必然打不开；唯一例外是
    # measured.md——指南规定实测结论以「实测得到」陈述并指向这份登记。
    for mention in RESEARCH_MENTION_RE.findall(text):
        if RESEARCH_ALLOWED_RE.match(mention):
            continue
        target = (path.parent / mention).resolve()
        if not target.exists():
            errors.append(
                f"引用了不存在的 research 文件：{mention}"
                "（改写成「实测得到」，或指向 research/measured.md）"
            )
        else:
            errors.append(
                f"引用了不发布的 research 路径：{mention}"
                "（research/ 不随站点发布，改写成「实测得到」，或指向 research/measured.md）"
            )

    parts = path.parts
    is_wiki_page = "wiki" in parts
    if is_wiki_page and path.name == "index.html":
        for name in REQUIRED_WIKI_META:
            if not inspector.meta.get(name):
                errors.append(f"缺少元数据：{name}")

        # 主题取值必须是封闭词表内的大类，防止主题再次碎片化
        topics = [
            item.strip()
            for item in re.split(r"[,，]", inspector.meta.get("dojo:topics", ""))
            if item.strip()
        ]
        for topic in topics:
            if topic not in ALLOWED_TOPICS:
                errors.append(
                    f"主题不在词表里：{topic}（可选：{'、'.join(ALLOWED_TOPICS)}）"
                )

        raw_status = inspector.meta.get("dojo:status", "")
        if raw_status and raw_status not in ALLOWED_STATUS:
            errors.append(
                f"状态不在词表里：{raw_status}（可选：{'、'.join(ALLOWED_STATUS)}；"
                "省略即为 current）"
            )

        raw_type = inspector.meta.get("dojo:type", "")
        if raw_type and raw_type not in ALLOWED_TYPES:
            errors.append(
                f"类型不在词表里：{raw_type}（可选：{'、'.join(ALLOWED_TYPES)}）"
            )
        elif raw_type and raw_type not in TYPE_STYLESHEET:
            errors.append(
                f"类型 {raw_type} 已暂停，只维护概念页"
                "（归档的页面要标 dojo:status=outdated）"
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
                    f"类型 {raw_type} 的页面要引用 "
                    + " 或 ".join(f"../../libs/{css}" for css in allowed_css)
                )

        # 标签取封闭词表内的单一值，防止细粒度标签再次碎片化
        raw_tag = inspector.meta.get("dojo:tag", "")
        if re.search(r"[,，]", raw_tag):
            errors.append(f"dojo:tag 只能写一个，现在是：{raw_tag}")
        elif raw_tag and raw_tag not in ALLOWED_TAGS:
            errors.append(
                f"标签不在词表里：{raw_tag}（可选：{'、'.join(ALLOWED_TAGS)}）"
            )

        description = inspector.meta.get("description", "")
        summary = inspector.meta.get("dojo:summary", "")
        if "$" in description:
            errors.append("description 是纯文本，公式写到 dojo:summary 里")
        if "$$" in summary:
            errors.append("dojo:summary 只支持行内公式 $...$")
        if summary.count("$") % 2:
            errors.append("dojo:summary 里的 $ 没有成对")

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
            errors.append("有公式的页面要引用本地的 KaTeX JS 和 CSS")
        if not has_auto_render:
            errors.append("有公式的页面要初始化 auto-render")

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
    按 outline-review-<轮次>.md、content-review-<轮次>.md 连续编号，资料放在 sources/ 下。
    """
    errors: list[str] = []
    if not research.is_dir():
        return errors
    for entry in sorted(research.iterdir()):
        if entry.is_dir():
            if entry.name not in RESEARCH_ALLOWED_DIRS:
                errors.append(f"research/ 里不该有目录：{entry.name}/")
            continue
        if not entry.name.endswith(".md"):
            errors.append(f"research/ 里不该有非 Markdown 文件：{entry.name}")
        elif entry.name in RESEARCH_ALLOWED_FILES or REVIEW_FILE_RE.fullmatch(entry.name):
            continue
        else:
            errors.append(f"research/ 里不该有这个文件：{entry.name}")
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
                f"<{context}> 里有没渲染的数学字符 {''.join(hits)}：{snippet}"
                "（用 $...$ 包起来，KaTeX 才会渲染）"
            )

    summary = inspector.meta.get("dojo:summary", "")
    hits = sorted(set(BARE_MATH_RE.findall(strip_math_segments(summary))))
    if hits:
        errors.append(
            f"dojo:summary 里有没渲染的数学字符 {''.join(hits)}"
            "（用 $...$ 包起来，KaTeX 才会渲染）"
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
                f"<{tag}> 里的数学定界符不会渲染：{snippet}"
                "（公式放进 <foreignObject>）"
            )
            continue
        hits = sorted(set(BARE_MATH_RE.findall(inner)))
        if hits:
            snippet = " ".join(inner.split())[:60]
            errors.append(
                f"<{tag}> 里有没渲染的数学字符 {''.join(hits)}：{snippet}"
                "（改用 <foreignObject> 加 $...$）"
            )
            continue
        for pattern in ASCII_MATH_PATTERNS:
            if pattern.search(inner):
                snippet = " ".join(inner.split())[:60]
                errors.append(
                    f"<{tag}> 里用 ASCII 近似写公式：{snippet}"
                    "（改用 <foreignObject> 加 $...$）"
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
                f"第 {index} 个 <pre> 块里是制表符画的图"
                f"（{len(hits)} 个制表符），改用 HTML 结构图或内联 SVG"
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
                    f"内联的 body margin 破坏了居中：margin: {value}"
                    "（写 margin: 0 auto，固定目录才不会盖住正文）"
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
        errors.append(f"模板的类型必须是 {module}，现在是：{raw_type or '（缺失）'}")

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
            "模板要么内联、要么引用共享样式表："
            + " 或 ".join(f"../../libs/{css}" for css in allowed_css)
        )
        return errors

    css_path = Path("libs") / canonical_css
    if not css_path.exists():
        errors.append(f"共享样式表不存在：{css_path}")
        return errors

    # 外链时没有副本，无从漂移，到此为止。
    if not blocks:
        return errors

    def normalize(css: str) -> str:
        css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
        return re.sub(r"\s+", " ", css).strip()

    if normalize(blocks[0]) != normalize(css_path.read_text(encoding="utf-8")):
        errors.append(
            f"模板内联的样式与 {css_path} 不一致，"
            "两份要保持逐字相同"
        )
    return errors


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

