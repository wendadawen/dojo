"""页面与模板的校验：HTML 结构、元数据、本地引用、数学字符、research/ 目录、模板，以及内联脚本语法。

由 validate.py 调度。
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
from vocab import ALLOWED_TAGS, ALLOWED_TOPICS  # noqa: E402


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
# 概念页要引用的共享样式表。
CONCEPT_STYLESHEET = "dojo-concept.css"
TEMPLATE_DIR = ".dojo/templates"
STYLE_RE = re.compile(r"<style[^>]*>(.*?)</style>", re.S | re.I)

# research/ 允许的文件：大纲、正文、资料冲突记录、各轮审查记录，以及 sources/。
RESEARCH_ALLOWED_FILES = {"outline.md", "content.md", "conflicts.md"}
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

# 页面正文里不能提到 research/ 下的路径：research/ 不随站点发布（CI 用 --exclude='research/' 排除），
# 读者点不开。
RESEARCH_MENTION_RE = re.compile(r"research/[\w./-]*[\w/-]")

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


def _links_stylesheet(inspector: "PageInspector") -> bool:
    """是否引用了共享样式表。部署会给链接加 ?v=<sha>，比对时忽略查询串。"""
    return any(
        ref.split("?", 1)[0].endswith(f"libs/{CONCEPT_STYLESHEET}")
        for ref in inspector.stylesheets
    )


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

    for mention in RESEARCH_MENTION_RE.findall(text):
        errors.append(f"页面里提到了不发布的路径：{mention}")

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

        if inspector.meta.get("dojo:type") != "concept":
            errors.append("dojo:type 必须是 concept")
        if not _links_stylesheet(inspector):
            errors.append(f"页面要引用 ../../libs/{CONCEPT_STYLESHEET}")

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
    """模板必须声明 dojo:type=concept，并引用共享样式表。模板含占位符，不能走 validate_page。"""
    errors: list[str] = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    inspector = PageInspector()
    inspector.feed(text)
    raw_type = inspector.meta.get("dojo:type", "")
    if raw_type != "concept":
        errors.append(f"模板的类型必须是 concept，现在是：{raw_type or '（缺失）'}")

    if not _links_stylesheet(inspector):
        errors.append(f"模板要引用 ../../libs/{CONCEPT_STYLESHEET}")
    elif not (Path("libs") / CONCEPT_STYLESHEET).exists():
        errors.append(f"共享样式表不存在：libs/{CONCEPT_STYLESHEET}")
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
    """收集页面里所有内联脚本的源码。"""

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

    只查语法，不执行代码。所有块在一个 node 进程里用 vm.Script 逐个编译。
    node 不存在时返回「未执行」，不阻断其他检查。
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

