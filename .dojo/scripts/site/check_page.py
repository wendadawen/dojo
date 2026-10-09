"""检查生成出来的页面（index.html、overview.html）。

每条规则是一个函数，收到解析好的页面，返回问题列表；增删规则要同时改 RULES 和 CASES。
只查生成器和模板保证不了的东西：正文可以原样写入 HTML，所以重复 id、没渲染的公式、手画的框线图都可能出现。
"""

from __future__ import annotations

import re
import tempfile
from html import unescape
from html.parser import HTMLParser
from pathlib import Path


LOCAL_REF_RE = re.compile(r'''(?:href|src)=["']([^"']+)["']''')
ID_RE = re.compile(r"""\bid=["']([^"']+)["']""")
ANCHOR_RE = re.compile(r"""href=["']#([^"']*)["']""")
SKIP_REF_PREFIXES = ("http://", "https://", "data:", "#", "mailto:")
# research/ 不随站点发布（CI 用 --exclude='research/' 排除），页面里提到它读者点不开
RESEARCH_MENTION_RE = re.compile(r"research/[\w./-]*[\w/-]")
CODE_TAGS = {"code", "pre", "samp", "kbd"}
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input",
             "link", "meta", "param", "source", "track", "wbr"}
UI_TAGS = {"button", "option", "title", "nav"}  # 界面控件里的字符不参与公式检查

INLINE_MATH_RE = re.compile(
    r"\$\$[\s\S]+?\$\$|"
    r"(?<!\\)\$(?!\$)(?:\\.|[^$\n])+?(?<!\\)\$(?!\$)|"
    r"\\\([\s\S]+?\\\)|"
    r"\\\[[\s\S]+?\\\]"
)
# 数学符号必须写成 LaTeX 交给 KaTeX 渲染，不能直接用 Unicode 字符。
# 只收必须由 KaTeX 排版的字符：希腊字母、上下标、数学运算符与关系符。
# × – → 等在中文技术散文里是普通排版字符，不列入。
BARE_MATH_RE = re.compile("[Α-Ωα-ω∂∇√∝∞∑∏∫≈≠≡≤≥≪≫∈∉⊂⊃∩∪∅⊕⊗⋅⁰-₟]")
# SVG 的 <text> 由浏览器直接绘制，KaTeX 不处理其中的 $...$，图里要公式得用 <foreignObject>
SVG_TEXT_RE = re.compile(r"<(text|tspan)\b[^>]*>(.*?)</\1>", re.S)
# 用 ASCII 近似写数学符号同样算没渲染：下标式（R_1）、写成单词的希腊字母、乘号的字符替代
ASCII_MATH_RES = (
    re.compile(r"\b[A-Za-z]\w*_\{?[A-Za-z0-9]"),
    re.compile(r"\b(?:alpha|beta|gamma|delta|theta|lambda|sigma|omega|phi|psi|mu|tau|epsilon|rho)\b", re.I),
    re.compile(r"\b(?:deg|degrees?|pi)\b", re.I),
    re.compile(r"[0-9A-Za-z]\s*\*\s*[0-9A-Za-z]"),
)
BOX_DRAWING_RE = re.compile(r"[─-╿▀-▟■-◿⬀-⯿]")
PRE_RE = re.compile(r"<pre\b[^>]*>([\s\S]*?)</pre>", re.I)
BOX_DRAWING_MIN_HITS = 4


# ---------- 解析 ----------

class Reader(HTMLParser):
    """读出页面里的正文文字（带所在标签）、meta 和 summary。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.prose: list[tuple[str, str]] = []  # (所在标签, 文字)，不含代码和脚本
        self._skip = 0
        self._code = 0
        self._stack: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self._skip += 1
        if tag in CODE_TAGS:
            self._code += 1
        if tag not in VOID_TAGS:
            self._stack.append(tag)
        if tag == "meta":
            values = dict(attrs)
            name = values.get("name", "").strip().lower()
            if name:
                self.meta[name] = values.get("content", "").strip()

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self._skip:
            self._skip -= 1
        if tag in CODE_TAGS and self._code:
            self._code -= 1
        if tag in self._stack:
            while self._stack and self._stack.pop() != tag:
                pass

    def handle_data(self, data):
        if self._skip or self._code or not data.strip():
            return
        self.prose.append((self._stack[-1] if self._stack else "body", data))


class Page:
    """一个页面，各条规则共用。"""

    def __init__(self, path: Path):
        self.path = path
        self.text = path.read_text(encoding="utf-8", errors="ignore")
        self.reader = Reader()
        self.reader.feed(self.text)


def without_math(text: str) -> str:
    return INLINE_MATH_RE.sub(" ", text)


# ---------- 规则：每条一个函数，返回问题列表 ----------

def ids_are_unique(p: Page) -> list[str]:
    seen, errors = set(), []
    for i in ID_RE.findall(p.text):
        if i in seen:
            errors.append(f"id 重复：{i}")
        seen.add(i)
    return errors


def anchors_exist(p: Page) -> list[str]:
    ids = set(ID_RE.findall(p.text))
    return [f"锚点指向不存在的 id：#{a}" for a in ANCHOR_RE.findall(p.text) if a and a not in ids]


def local_refs_exist(p: Page) -> list[str]:
    errors = []
    for ref in LOCAL_REF_RE.findall(p.text):
        if ref.startswith(SKIP_REF_PREFIXES):
            continue
        clean = ref.split("#", 1)[0].split("?", 1)[0]
        if not clean:
            continue
        target = (p.path.parent / clean).resolve()
        if clean.endswith("/"):
            target = target / "index.html"
        if not target.exists():
            errors.append(f"本地引用不存在：{ref}")
    return errors


def no_research_paths(p: Page) -> list[str]:
    return [f"页面里提到了不发布的路径：{m}" for m in RESEARCH_MENTION_RE.findall(p.text)]


def no_bare_math_chars(p: Page) -> list[str]:
    """公式定界符之外出现数学 Unicode 字符，就是没渲染的公式。覆盖标题、summary、正文与列表，不含代码块。"""
    errors = []
    for tag, chunk in p.reader.prose:
        if tag in UI_TAGS:
            continue
        outside = without_math(chunk)
        hits = sorted(set(BARE_MATH_RE.findall(outside)))
        if hits:
            errors.append(f"<{tag}> 里有没渲染的数学字符 {''.join(hits)}：{' '.join(outside.split())[:70]}"
                          "（用 $...$ 包起来，KaTeX 才会渲染）")
    hits = sorted(set(BARE_MATH_RE.findall(without_math(p.reader.meta.get("dojo:summary", "")))))
    if hits:
        errors.append(f"dojo:summary 里有没渲染的数学字符 {''.join(hits)}（用 $...$ 包起来，KaTeX 才会渲染）")
    return errors


def svg_text_has_no_math(p: Page) -> list[str]:
    errors = []
    for tag, content in SVG_TEXT_RE.findall(p.text):
        inner = unescape(re.sub(r"<[^>]+>", "", content))
        snippet = " ".join(inner.split())[:60]
        if INLINE_MATH_RE.search(inner):
            errors.append(f"<{tag}> 里的数学定界符不会渲染：{snippet}（公式放进 <foreignObject>）")
        elif BARE_MATH_RE.search(inner):
            hits = "".join(sorted(set(BARE_MATH_RE.findall(inner))))
            errors.append(f"<{tag}> 里有没渲染的数学字符 {hits}：{snippet}（改用 <foreignObject> 加 $...$）")
        elif any(r.search(inner) for r in ASCII_MATH_RES):
            errors.append(f"<{tag}> 里用 ASCII 近似写公式：{snippet}（改用 <foreignObject> 加 $...$）")
    return errors


def pre_has_no_box_drawing(p: Page) -> list[str]:
    errors = []
    for index, block in enumerate(PRE_RE.findall(p.text), start=1):
        hits = BOX_DRAWING_RE.findall(block)
        if len(hits) >= BOX_DRAWING_MIN_HITS:
            errors.append(f"第 {index} 个 <pre> 块里是制表符画的图（{len(hits)} 个制表符），改用 HTML 结构图或内联 SVG")
    return errors


RULES = [
    ids_are_unique, anchors_exist, local_refs_exist, no_research_paths,
    no_bare_math_chars, svg_text_has_no_math, pre_has_no_box_drawing,
]


def check(path: Path) -> list[str]:
    p = Page(path)
    errors: list[str] = []
    for rule in RULES:
        errors += rule(p)
    return errors


# ---------- 自检 ----------

WRAP = "<!DOCTYPE html><html><head><title>t</title>{head}</head><body>{body}</body></html>"
CASES = [
    ("干净的页面没有报错", "", "<p>正文 $x^2$。</p>", None),
    ("id 重复", "", '<h2 id="a">1</h2><h2 id="a">2</h2>', "id 重复"),
    ("锚点指向不存在的 id", "", '<a href="#nope">跳</a>', "锚点指向不存在"),
    ("本地引用不存在", "", '<img src="assets/nope.png">', "本地引用不存在"),
    ("提到 research 路径", "", "<p>见 research/outline.md</p>", "不发布的路径"),
    ("正文里的希腊字母没包公式", "", "<p>参数 α 取值</p>", "没渲染的数学字符"),
    ("代码块里的希腊字母不查", "", "<pre><code>α = 1</code></pre>", None),
    ("summary 里的希腊字母没包公式", '<meta name="dojo:summary" content="α">', "<p>x</p>", "dojo:summary"),
    ("SVG 文字里写公式", "", "<svg><text>$x$</text></svg>", "数学定界符不会渲染"),
    ("SVG 文字里写 ASCII 公式", "", "<svg><text>theta_i</text></svg>", "ASCII 近似"),
    ("<pre> 里画框线图", "", "<pre>┌──┐│ ││ ││ │└──┘</pre>", "制表符画的图"),
]


def selftest() -> list[tuple[str, bool]]:
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "index.html"
        for name, head, body, want in CASES:
            path.write_text(WRAP.format(head=head, body=body), encoding="utf-8")
            errs = check(path)
            results.append((name, not errs if want is None else any(want in e for e in errs)))
    return results
