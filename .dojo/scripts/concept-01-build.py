#!/usr/bin/env python3
"""概念页的生成器：读 research/concept.md，生成 index.html 和 overview.html。

用法（从仓库根运行）：
    python3 .dojo/scripts/concept-01-build.py wiki/<name>/research/concept.md
    python3 .dojo/scripts/concept-01-build.py wiki/<name>/research/concept.md --run
    python3 .dojo/scripts/concept-01-build.py --selftest

文件格式见 guides/concept/write.md。页面上的文字都来自 concept.md，生成器只做装配：
把 Markdown 转成 HTML，给章节编号，生成核心问题的折叠块和文末的参考资料。
--run 另外运行后面紧跟 text 代码块的 Python 代码，逐行比对输出；不跟输出块的算源码摘录，不运行。

它不做这些事：
  · 不定义页面骨架——骨架是 .dojo/templates/concept/ 下的 index.html 和 overview.html
  · 不校验事实——那是审查的事

只认下面这些写法，写了别的会报错，不会静默产出错页面：
  段落、**加粗**、`代码`、[链接](地址)、![图](assets/…)、$行内公式$、$$独立公式$$、
  - 无序列表、1. 有序列表、表格、``` 代码块、### 小节标题、
  「::: 补充：/展开：/代码：<标题>」到「:::」的折叠块、<figure>/<div>/<svg> 开头的原样 HTML。
"""

from __future__ import annotations

import html
import itertools
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE.parent / "templates" / "concept"
REPO_ROOT = HERE.parents[1]

META_KEYS = {"描述": "description", "摘要": "summary", "主题": "topics", "标签": "tag"}
# 固定小节：不是正文章节。其余 ## 标题按出现顺序编号成章。
RESERVED = ("资料", "范围", "依据", "概览", "引言", "核心问题", "常见误解")
REQUIRED = ("资料", "范围", "依据", "概览", "引言", "核心问题")
GENERATED = ("参考资料", "本章问题")
DETAIL_PREFIXES = ("补充：", "展开：", "代码：")

MATH_RE = re.compile(r"\$\$.+?\$\$|(?<!\\)\$(?!\$)(?:\\.|[^$\n])+?(?<!\\)\$", re.S)
LINE_REF_RE = re.compile(r"\.(?:py|pyi|cu|cuh|cc|cpp|c|h|hpp|rs|go|js|ts|java)(?::|#L)\d+")
LIST_RE = re.compile(r"^(\s*)(-|\d+\.)\s+(.*)$")
RAW_HTML_RE = re.compile(r"^<(figure|div|svg)\b")
NUMBERED_RE = re.compile(r"^\d+(?:\.\d+)*[.、．\s]")
CJK_RE = re.compile(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]")

Line = tuple[int, str]


def fail(no: int, msg: str) -> None:
    raise SystemExit(f"第 {no} 行：{msg}" if no else msg)


class Ctx:
    """一次生成的上下文：链接按输出目录解析，代码块按出现顺序记下供 --run 使用。"""

    def __init__(self, out_dir: Path, wiki_root: Path) -> None:
        self.out_dir = out_dir
        self.wiki_root = wiki_root
        self.fences: list[tuple[int, str, str]] = []
        self.details = 0


# ---------- 行内 ----------

def _link(label: str, href: str, no: int, ctx: Ctx) -> str:
    raw = html.unescape(href)
    if raw.startswith(("http://", "https://", "#")):
        target = raw
    elif raw.startswith("wiki/"):
        path, _, anchor = raw.partition("#")
        m = re.fullmatch(r"wiki/([\w.-]+)(?:/(index\.html|overview\.html)?)?", path)
        if not m:
            fail(no, f"站内链接写成 wiki/<页面名>：{raw}")
        page = ctx.wiki_root / m.group(1) / (m.group(2) or "index.html")
        if not page.exists():
            fail(no, f"链接的页面不存在：{raw}")
        target = os.path.relpath(page, ctx.out_dir).replace(os.sep, "/")
        target += f"#{anchor}" if anchor else ""
    elif raw.startswith("assets/"):
        if not (ctx.out_dir / raw).exists():
            fail(no, f"链接的文件不存在：{raw}")
        target = raw
    else:
        fail(no, f"链接只能写 http(s) 地址、wiki/<页面名> 或 assets/<文件>：{raw}")
    return f'<a href="{html.escape(target)}">{label}</a>'


def _img(alt: str, src: str, no: int, ctx: Ctx) -> str:
    raw = html.unescape(src)
    if not raw.startswith("assets/") or not (ctx.out_dir / raw).exists():
        fail(no, f"图片放在页面的 assets/ 下，且文件要存在：{raw}")
    return f'<img src="{html.escape(raw)}" alt="{alt}">'


def inline(text: str, no: int, ctx: Ctx) -> str:
    slots: list[str] = []

    def keep(s: str) -> str:
        slots.append(s)
        return f"\x00{len(slots) - 1}\x00"

    text = MATH_RE.sub(lambda m: keep(html.escape(m.group(0), quote=False)), text)
    if "$" in text.replace("\\$", ""):
        fail(no, f"有落单的 $：{text.strip()[:40]}")
    text = re.sub(r"`([^`]+)`",
                  lambda m: keep(f"<code>{html.escape(m.group(1), quote=False)}</code>"), text)
    text = html.escape(text, quote=False)
    text = re.sub(r"!\[([^\]]*)\]\(([^)\s]+)\)",
                  lambda m: keep(_img(m.group(1), m.group(2), no, ctx)), text)
    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)",
                  lambda m: keep(_link(m.group(1), m.group(2), no, ctx)), text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    while "\x00" in text:
        text = re.sub(r"\x00(\d+)\x00", lambda m: slots[int(m.group(1))], text)
    return text


def _join(parts: list[str]) -> str:
    """段落内换行：中文之间直接相连，西文之间补一个空格。"""
    out = ""
    for p in parts:
        if out and not (CJK_RE.match(out[-1]) or CJK_RE.match(p[0])):
            out += " "
        out += p
    return out


# ---------- 块 ----------

def _is_block_start(raw: str) -> bool:
    s = raw.strip()
    return (s.startswith(("```", "$$", ":::", "|", "#", ">"))
            or bool(RAW_HTML_RE.match(s)) or bool(LIST_RE.match(raw)))


def _fence(lines: list[Line], i: int, ctx: Ctx, out: list[str]) -> int:
    no, raw = lines[i]
    indent = len(raw) - len(raw.lstrip())
    lang = raw.strip()[3:].strip() or "text"
    if not re.fullmatch(r"[\w+-]+", lang):
        fail(no, f"代码块的语言写错了：{lang}")
    body, j = [], i + 1
    while j < len(lines) and lines[j][1].strip() != "```":
        r = lines[j][1]
        body.append(r[indent:] if not r[:indent].strip() else r)
        j += 1
    if j >= len(lines):
        fail(no, "代码块缺少结尾的 ```")
    code = "\n".join(body)
    ctx.fences.append((no, lang, code))
    out.append(f'<div class="code-block"><pre><code class="language-{lang}">'
               f"{html.escape(code, quote=False)}</code></pre></div>")
    return j + 1


def _display_math(lines: list[Line], i: int, out: list[str]) -> int:
    no, raw = lines[i]
    s = raw.strip()
    if len(s) > 4 and s.endswith("$$"):
        out.append(f"<p>{html.escape(s, quote=False)}</p>")
        return i + 1
    buf, j = [s], i + 1
    while j < len(lines):
        t = lines[j][1].strip()
        buf.append(t)
        if t.endswith("$$"):
            break
        j += 1
    else:
        fail(no, "独立公式缺少结尾的 $$")
    out.append("<p>" + html.escape("\n".join(buf), quote=False) + "</p>")
    return j + 1


def _container(lines: list[Line], i: int, ctx: Ctx, out: list[str]) -> int:
    no, raw = lines[i]
    title = raw.strip()[3:].strip()
    if not title.startswith(DETAIL_PREFIXES) or len(title) <= 3:
        fail(no, "折叠块写成「::: 补充：标题」「::: 展开：标题」或「::: 代码：标题」")
    depth, j, fence = 1, i + 1, False
    while j < len(lines):
        t = lines[j][1].strip()
        if t.startswith("```"):
            fence = not fence
        elif not fence and t == ":::":
            depth -= 1
            if depth == 0:
                break
        elif not fence and t.startswith(":::"):
            depth += 1
        j += 1
    else:
        fail(no, "折叠块缺少结尾的 :::")
    inner = render_blocks(lines[i + 1:j], ctx)
    if not inner:
        fail(no, "折叠块是空的")
    cls = ' class="code-details"' if title.startswith("代码：") else ""
    out.append(f"<details{cls}>\n<summary>{inline(title, no, ctx)}</summary>\n"
               + "\n".join(inner) + "\n</details>")
    ctx.details += 1
    return j + 1


def _raw_html(lines: list[Line], i: int, out: list[str]) -> int:
    no, raw = lines[i]
    tag = RAW_HTML_RE.match(raw.strip()).group(1)
    opening, closing = re.compile(rf"<{tag}\b"), re.compile(rf"</{tag}>")
    depth, j, buf = 0, i, []
    while j < len(lines):
        t = lines[j][1]
        buf.append(t)
        depth += len(opening.findall(t)) - len(closing.findall(t))
        if depth <= 0:
            break
        j += 1
    else:
        fail(no, f"<{tag}> 没有闭合")
    out.append("\n".join(buf))
    return j + 1


def split_row(line: str) -> list[str]:
    """表格一行拆成单元格：公式里的 | 不拆，公式外的 \\| 是字面竖线。"""
    s = line.strip()
    s = s[1:] if s.startswith("|") else s
    s = s[:-1] if s.endswith("|") and not s.endswith("\\|") else s
    cells, buf, in_math, k = [], "", False, 0
    while k < len(s):
        c = s[k]
        if c == "\\" and s[k + 1:k + 2] == "|":
            buf += "\\|" if in_math else "|"
            k += 2
            continue
        if c == "$":
            in_math = not in_math
        if c == "|" and not in_math:
            cells.append(buf.strip())
            buf = ""
        else:
            buf += c
        k += 1
    cells.append(buf.strip())
    return cells


def _table(lines: list[Line], i: int, ctx: Ctx, out: list[str]) -> int:
    rows, j = [], i
    while j < len(lines) and lines[j][1].strip().startswith("|"):
        rows.append(lines[j])
        j += 1
    seps = split_row(rows[1][1]) if len(rows) > 1 else []
    if not seps or not all(re.fullmatch(r":?-{3,}:?", c) for c in seps):
        fail(rows[0][0], "表格第二行要是 |---|---| 分隔行")
    head = split_row(rows[0][1])
    if len(seps) != len(head):
        fail(rows[1][0], "分隔行的列数和表头不一样")
    align = ["right" if c.endswith(":") and not c.startswith(":")
             else "center" if c.startswith(":") and c.endswith(":") else "" for c in seps]

    def cell(tag: str, text: str, k: int, no: int) -> str:
        style = f' style="text-align:{align[k]}"' if align[k] else ""
        return f"<{tag}{style}>{inline(text, no, ctx)}</{tag}>"

    th = "".join(cell("th", c, k, rows[0][0]) for k, c in enumerate(head))
    trs = []
    for no, r in rows[2:]:
        cells = split_row(r)
        if len(cells) != len(head):
            fail(no, f"这一行有 {len(cells)} 列，表头有 {len(head)} 列")
        trs.append("<tr>" + "".join(cell("td", c, k, no) for k, c in enumerate(cells)) + "</tr>")
    out.append('<div class="table-scroll">\n<table>\n'
               f"<thead><tr>{th}</tr></thead>\n<tbody>\n" + "\n".join(trs)
               + "\n</tbody>\n</table>\n</div>")
    return j


def _list(lines: list[Line], i: int, ctx: Ctx, out: list[str]) -> int:
    m = LIST_RE.match(lines[i][1])
    base, ordered = len(m.group(1)), m.group(2) != "-"
    items: list[list[Line]] = []
    content_indent = 0
    j = i
    while j < len(lines):
        no, r = lines[j]
        mm = LIST_RE.match(r)
        if not r.strip():
            k = j + 1
            while k < len(lines) and not lines[k][1].strip():
                k += 1
            if k < len(lines):
                nxt = lines[k][1]
                mk = LIST_RE.match(nxt)
                ind = len(nxt) - len(nxt.lstrip())
                same = mk and len(mk.group(1)) == base and (mk.group(2) != "-") == ordered
                if same or ind > base:
                    items[-1].append((no, ""))
                    j += 1
                    continue
            break
        ind = len(r) - len(r.lstrip())
        if mm and ind == base:
            if (mm.group(2) != "-") != ordered:
                break
            content_indent = base + len(mm.group(2)) + 1
            items.append([(no, mm.group(3))])
        elif ind > base:
            items[-1].append((no, r[min(ind, content_indent):]))
        elif not _is_block_start(r):
            items[-1].append((no, r.strip()))
        else:
            break
        j += 1
    lis = []
    for item in items:
        inner = render_blocks(item, ctx)
        if len(inner) == 1 and inner[0].startswith("<p>") and inner[0].endswith("</p>"):
            lis.append(f"<li>{inner[0][3:-4]}</li>")
        else:
            lis.append("<li>\n" + "\n".join(inner) + "\n</li>")
    tag = "ol" if ordered else "ul"
    out.append(f"<{tag}>\n" + "\n".join(lis) + f"\n</{tag}>")
    return j


def _paragraph(lines: list[Line], i: int, ctx: Ctx, out: list[str]) -> int:
    buf, j = [], i
    while j < len(lines):
        r = lines[j][1]
        if not r.strip() or (j > i and _is_block_start(r)):
            break
        buf.append(r.strip())
        j += 1
    no, text = lines[i][0], _join(buf)
    m = re.fullmatch(r"!\[([^\]]*)\]\(([^)\s]+)\)", text)
    out.append(_img(html.escape(m.group(1)), m.group(2), no, ctx) if m
               else f"<p>{inline(text, no, ctx)}</p>")
    return j


def render_blocks(lines: list[Line], ctx: Ctx, heading=None) -> list[str]:
    """一段 Markdown 行转成 HTML 块。heading 为 None 时不允许出现标题。"""
    out: list[str] = []
    i = 0
    while i < len(lines):
        no, raw = lines[i]
        s = raw.strip()
        if not s:
            i += 1
        elif s.startswith("```"):
            i = _fence(lines, i, ctx, out)
        elif s.startswith("$$"):
            i = _display_math(lines, i, out)
        elif s.startswith(":::"):
            i = _container(lines, i, ctx, out)
        elif RAW_HTML_RE.match(s):
            i = _raw_html(lines, i, out)
        elif s.startswith("#"):
            m = re.fullmatch(r"###\s+(.+)", s)
            if heading is None or not m:
                fail(no, "这里不能写标题" if heading is None else "小节标题用 ###")
            out.append(heading(m.group(1).strip(), no))
            i += 1
        elif s.startswith("|"):
            i = _table(lines, i, ctx, out)
        elif LIST_RE.match(raw):
            i = _list(lines, i, ctx, out)
        elif s.startswith(">") or re.fullmatch(r"[-*_]{3,}", s):
            fail(no, "不支持引用块和分隔线")
        else:
            i = _paragraph(lines, i, ctx, out)
    return out


# ---------- 整份文件 ----------

def parse(text: str) -> dict:
    """按 guides/concept/write.md 的格式切分：标题、页面信息、各小节的原始行。"""
    title, meta = "", {}
    sections: dict[str, list[Line]] = {}
    order: list[Line] = []
    current, fence = None, False
    for no, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if s.startswith("```"):
            fence = not fence
        if not fence and line.startswith("## "):
            name = line[3:].strip()
            if name in GENERATED:
                fail(no, f"「{name}」由生成器处理，不要自己写")
            if name in sections:
                fail(no, f"「{name}」重复了")
            current = name
            sections[name] = []
            order.append((no, name))
        elif current is not None:
            sections[current].append((no, line))
        elif line.startswith("# "):
            if title:
                fail(no, "只能有一个 # 标题")
            title = line[2:].strip()
        elif s:
            m = re.match(r"-\s*([^：:]+)[：:]\s*(.*)", s)
            if not title or not m or m.group(1).strip() not in META_KEYS:
                fail(no, "开头先写「# 标题」，下面只写 描述、摘要、主题、标签 四行")
            meta[META_KEYS[m.group(1).strip()]] = m.group(2).strip()

    if not title:
        fail(0, "缺少「# 标题」")
    if "$" in title:
        fail(0, "标题不写公式")
    for zh, key in META_KEYS.items():
        if not meta.get(key):
            fail(0, f"缺少页面信息：{zh}")
    if "$" in meta["description"]:
        fail(0, "描述是纯文本，公式写到摘要里")
    for name in REQUIRED:
        if not any(l.strip() for _, l in sections.get(name, [])):
            fail(0, f"缺少「{name}」一节")
    chapters = [(no, name) for no, name in order if name not in RESERVED]
    if not chapters:
        fail(0, "没有正文章节")
    for no, name in chapters:
        if NUMBERED_RE.match(name):
            fail(no, "章节标题不写编号，生成器会编号")

    refs = []
    for no, line in sections["资料"]:
        s = line.strip()
        if not s:
            continue
        if not s.startswith("- "):
            fail(no, "资料每条写成「- 」开头的一行")
        refs.append((no, s[2:].strip()))

    basis = [l for l in sections["依据"] if l[1].strip()]
    if not basis or not basis[0][1].strip().startswith("|"):
        fail(sections["依据"][0][0] if sections["依据"] else 0, "依据写成表格：| 论断 | 出处 |")
    head = split_row(basis[0][1])
    if head[:2] != ["论断", "出处"] or len(basis) < 3:
        fail(basis[0][0], "依据的表头是 | 论断 | 出处 |，下面至少一行")

    rendered = ["资料", "概览", "引言", "核心问题", "常见误解"] + [n for _, n in chapters]
    fence = False
    for name in rendered:
        for no, line in sections.get(name, []):
            if line.strip().startswith("```"):
                fence = not fence
            elif not fence and LINE_REF_RE.search(line):
                fail(no, "页面上不写源码行号，行号只写在「依据」里")
    for key in ("description", "summary"):
        if LINE_REF_RE.search(meta[key]):
            fail(0, "页面上不写源码行号，行号只写在「依据」里")

    return {"title": title, "meta": meta, "sections": sections,
            "chapters": chapters, "refs": refs}


def _core_questions(lines: list[Line], ctx: Ctx) -> str:
    qs: list[tuple[int, str, list[Line]]] = []
    for no, line in lines:
        s = line.strip()
        if s.startswith("### "):
            qs.append((no, s[4:].strip(), []))
        elif qs:
            qs[-1][2].append((no, line))
        elif s:
            fail(no, "核心问题一节里，每个问题写成「### 问题」")
    if not 3 <= len(qs) <= 5:
        fail(lines[0][0], f"核心问题要 3 到 5 个，现在是 {len(qs)} 个")
    items = []
    for no, q, body in qs:
        k = next((n for n, (_, l) in enumerate(body) if l.strip()), None)
        if k is None or not body[k][1].strip().startswith("解答："):
            fail(no, "问题下面第一行写「解答：<一句话要点>」")
        rest = render_blocks(body[k + 1:], ctx)
        if not rest:
            fail(body[k][0], "解答要点下面还要写完整答案")
        items.append(f"<li>\n<p>{inline(q, no, ctx)}</p>\n<details>\n"
                     f"<summary>{inline(body[k][1].strip(), body[k][0], ctx)}</summary>\n"
                     + "\n".join(rest) + "\n</details>\n</li>")
    return ('<section class="learning-goals">\n<h2>核心问题</h2>\n<ol class="chapter-questions">\n'
            + "\n".join(items) + "\n</ol>\n</section>")


def render(doc: dict, ctx: Ctx) -> tuple[str, str, dict]:
    sec = doc["sections"]
    parts = render_blocks(sec["引言"], ctx)
    parts.append(_core_questions(sec["核心问题"], ctx))
    if any(l.strip() for _, l in sec.get("常见误解", [])):
        parts.append('<section class="misconceptions">\n<h2>常见误解</h2>\n'
                     + "\n".join(render_blocks(sec["常见误解"], ctx)) + "\n</section>")
    for k, (no, name) in enumerate(doc["chapters"], 1):
        parts.append(f'<h2 id="s{k}">{k}. {inline(name, no, ctx)}</h2>')
        counter = itertools.count(1)

        def sub(text: str, line_no: int, k: int = k, counter=counter) -> str:
            j = next(counter)
            return f'<h3 id="s{k}-{j}">{k}.{j} {inline(text, line_no, ctx)}</h3>'

        parts += render_blocks(sec[name], ctx, heading=sub)
    parts.append('<h2 id="references">参考资料</h2>\n<ol>\n'
                 + "\n".join(f"<li>{inline(t, no, ctx)}</li>" for no, t in doc["refs"])
                 + "\n</ol>")
    overview = render_blocks(sec["概览"], ctx,
                             heading=lambda t, no: f"<h2>{inline(t, no, ctx)}</h2>")
    stats = {"chapters": len(doc["chapters"]), "details": ctx.details}
    return "\n\n".join(parts), "\n\n".join(overview), stats


def _fill(tpl: str, out_dir: Path, pairs: list[tuple[str, str]]) -> str:
    target = out_dir.resolve()
    if REPO_ROOT in target.parents:
        up = os.path.relpath(REPO_ROOT, target).replace(os.sep, "/")
        tpl = tpl.replace('"../../', f'"{up}/')
    for old, new in pairs:
        if old not in tpl:
            raise SystemExit(f"模板里找不到「{old[:30]}」，模板改了？")
        tpl = tpl.replace(old, new) if not old.startswith("<!--") else tpl.replace(old, new, 1)
    leftover = re.findall(r"【[^】]*】", tpl)
    if leftover:
        raise SystemExit(f"占位符未替换完，模板改了？{leftover}")
    return tpl


def build(md: Path, out_dir: Path | None = None, wiki_root: Path | None = None,
          templates: Path | None = None) -> tuple[Path, Path, dict, Ctx]:
    """research/concept.md → 页面目录下的 index.html 和 overview.html。"""
    if out_dir is None:
        if md.parent.name != "research":
            raise SystemExit(f"{md} 不在 research/ 下，用 -o 指定输出目录")
        out_dir = md.parent.parent
    ctx = Ctx(out_dir, wiki_root or REPO_ROOT / "wiki")
    doc = parse(md.read_text(encoding="utf-8"))
    body, overview, stats = render(doc, ctx)
    meta, title = doc["meta"], doc["title"]
    short = title.split("：", 1)[0].strip()
    tdir = templates or TEMPLATES

    index_tpl = (tdir / "index.html").read_text(encoding="utf-8")
    index_tpl = re.sub(r"<style>[\s\S]*?</style>",
                       lambda _: '<link rel="stylesheet" href="../../libs/dojo-concept.css">',
                       index_tpl, count=1)
    content_marker = re.search(r"<!-- @content[\s\S]*?-->", index_tpl)
    if not content_marker:
        raise SystemExit("index.html 模板里没有 @content 标记")
    index_html = _fill(index_tpl, out_dir, [
        (content_marker.group(0), body),
        ("【首页摘要】", html.escape(meta["description"])),
        ("【首页可渲染摘要；公式使用 $...$】", html.escape(meta["summary"])),
        ("【主题，逗号分隔】", html.escape(meta["topics"])),
        ("【内容标签】", html.escape(meta["tag"])),
        ("【概念名】：【简要说明核心作用】", html.escape(title)),
        ("【概念名】", html.escape(short)),
    ])

    overview_tpl = (tdir / "overview.html").read_text(encoding="utf-8")
    ov_marker = re.search(r"<!-- @content -->(?:\s*<!--[\s\S]*?-->)?", overview_tpl)
    if not ov_marker:
        raise SystemExit("overview.html 模板里没有 @content 标记")
    overview_html = _fill(overview_tpl, out_dir, [
        (ov_marker.group(0), overview),
        ("【概念名】", html.escape(short)),
        ("【主题标签】", html.escape(meta["tag"])),
        ("【定位摘要：这个概念是什么、解决什么问题】", html.escape(meta["description"])),
    ])

    out_dir.mkdir(parents=True, exist_ok=True)
    index_path, overview_path = out_dir / "index.html", out_dir / "overview.html"
    index_path.write_text(index_html, encoding="utf-8")
    overview_path.write_text(overview_html, encoding="utf-8")
    return index_path, overview_path, stats, ctx


def run_code(fences: list[tuple[int, str, str]]) -> tuple[int, list[str]]:
    """运行后面紧跟 text 代码块的 python 代码，把 text 块当预期输出逐行比对。

    后面不跟输出块的 python 代码是源码摘录，不运行。
    """
    problems, ran = [], 0
    for k, (no, lang, code) in enumerate(fences):
        nxt = fences[k + 1] if k + 1 < len(fences) else None
        if lang != "python" or not nxt or nxt[1] != "text":
            continue
        expected = nxt[2]
        with tempfile.TemporaryDirectory() as tmp:
            try:
                proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                      text=True, timeout=600, cwd=tmp)
            except subprocess.TimeoutExpired:
                problems.append(f"第 {no} 行的代码运行超过 10 分钟")
                continue
        ran += 1
        if proc.returncode != 0:
            last = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""
            problems.append(f"第 {no} 行的代码运行失败（退出码 {proc.returncode}）：{last}")
        else:
            got = [l.rstrip() for l in proc.stdout.rstrip().splitlines()]
            want = [l.rstrip() for l in expected.rstrip().splitlines()]
            if got != want:
                problems.append(f"第 {no} 行的代码输出和后面的输出块不一致\n  实际：{got}\n  文中：{want}")
    return ran, problems


# ---------- 自检 ----------

def _selftest() -> int:
    """用一份与任何真实概念无关的数据跑一遍，确认生成器可用、该报错的地方会报错。"""
    doc = "\n".join([
        "# 玩具算子：自检用的虚构概念",
        "",
        "- 描述：生成器自检页。",
        "- 摘要：自检，与任何真实概念无关；公式 $y=2x$。",
        "- 主题：数学基础",
        "- 标签：数学与数值",
        "",
        "## 资料",
        "",
        "- 虚构论文：[示例](https://example.com/paper)",
        "",
        "## 范围",
        "",
        "- 讲：RANGE-ONLY 玩具算子",
        "",
        "## 依据",
        "",
        "| 论断 | 出处 |",
        "|---|---|",
        "| 输出是输入的两倍 | toy.py:12 BASIS-ONLY |",
        "",
        "## 概览",
        "",
        "### 是什么",
        "",
        "玩具算子把输入乘 2，前置见[前置页](wiki/prereq)。",
        "",
        "## 引言",
        "",
        "开头一段，",
        "换行接着写。",
        "",
        "## 核心问题",
        "",
        "### 它做什么？",
        "",
        "解答：乘 2",
        "",
        "输出 $y=2x$，见第一章。",
        "",
        "### 为什么？",
        "",
        "解答：自检",
        "",
        "因为要自检。",
        "",
        "### 何时不成立？",
        "",
        "解答：从不",
        "",
        "自检里总成立。",
        "",
        "## 定义",
        "",
        "公式如下：",
        "$$",
        "y = 2x,\\qquad a<b",
        "$$",
        "",
        "- $y$：输出",
        "- 嵌套：",
        "  - 第二层 **加粗**",
        "",
        "| 输入 | 输出 |",
        "|---|---:|",
        "| $1$ | $2$ |",
        "| $\\|x\\|$ | a\\|b |",
        "",
        "### 小节",
        "",
        "::: 代码：验证乘 2",
        "```python",
        "print(2 * 1)",
        "```",
        "",
        "```text",
        "2",
        "```",
        ":::",
        "",
        "## 边界",
        "",
        "源码摘录不运行：",
        "",
        "```python",
        "def forward(self, x):",
        "    return self.missing(x)",
        "```",
        "",
        "::: 补充：更多",
        "补充内容。",
        ":::",
    ])

    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        wiki = root / "wiki"
        (wiki / "prereq").mkdir(parents=True)
        (wiki / "prereq" / "index.html").write_text("<html></html>", encoding="utf-8")
        pdir = root / "wiki" / "toy" / "research"
        pdir.mkdir(parents=True)
        md = pdir / "concept.md"
        md.write_text(doc, encoding="utf-8")
        idx, ov, stats, ctx = build(md, wiki_root=wiki)
        t, o = idx.read_text(encoding="utf-8"), ov.read_text(encoding="utf-8")
        checks += [
            ("输出到页面目录", idx == root / "wiki" / "toy" / "index.html" and ov.exists()),
            ("无占位符与标记残留", "【" not in t + o and "@content" not in t + o),
            ("样式改为外链", "dojo-concept.css" in t and "<style>" not in t),
            ("标题与导航", "<h1 class=\"title\">玩具算子：自检用的虚构概念</h1>" in t
             and '<span class="nav-brand">玩具算子</span>' in t),
            ("章节编号", '<h2 id="s1">1. 定义</h2>' in t and '<h2 id="s2">2. 边界</h2>' in t),
            ("小节编号", '<h3 id="s1-1">1.1 小节</h3>' in t),
            ("核心问题带解答", t.count("<summary>解答：") == 3 and 'class="learning-goals"' in t),
            ("代码折叠块", '<details class="code-details">' in t and "language-python" in t),
            ("普通折叠块", "<summary>补充：更多</summary>" in t),
            ("公式原样保留、< 转义", "y = 2x,\\qquad a&lt;b" in t),
            ("表格右对齐", 'style="text-align:right"' in t),
            ("表格里公式的竖线不拆", "$\\|x\\|$" in t and "a|b" in t),
            ("嵌套列表", t.count("<ul>") == 2 and "<b>加粗</b>" in t),
            ("中文换行不留空格", "开头一段，换行接着写。" in t),
            ("参考资料", '<h2 id="references">参考资料</h2>' in t
             and 'href="https://example.com/paper"' in t),
            ("范围和依据不进页面", "RANGE-ONLY" not in t + o and "BASIS-ONLY" not in t + o
             and "toy.py" not in t + o),
            ("概览：小标题与站内链接", "<h2>是什么</h2>" in o and 'href="../prereq/index.html"' in o),
            ("概览：摘要进 lead", '<p class="lead">生成器自检页。</p>' in o),
            ("统计", stats == {"chapters": 2, "details": 2}),
        ]
        ran, problems = run_code(ctx.fences)
        checks.append(("--run：只跑带输出块的代码，输出一致", ran == 1 and not problems))
        ran, problems = run_code([(1, "python", "print(3)"), (2, "text", "2")])
        checks.append(("--run：输出不一致会报出", ran == 1 and len(problems) == 1))

        again, _, _, _ = build(md, wiki_root=wiki)
        checks.append(("幂等", again.read_text(encoding="utf-8") == t))

        for name, broken in [
            ("缺少范围", doc.replace("## 范围", "## 其他章").replace("- 讲：RANGE-ONLY 玩具算子", "文字")),
            ("正文写行号", doc.replace("补充内容。", "见 toy.py:12。")),
            ("章节标题写编号", doc.replace("## 定义", "## 1. 定义")),
            ("缺少标签", doc.replace("- 标签：数学与数值\n", "")),
            ("核心问题不足 3 个", doc.replace("### 何时不成立？", "何时不成立？")),
            ("解答缺要点行", doc.replace("解答：乘 2", "乘 2")),
            ("站内链接指向不存在的页面", doc.replace("wiki/prereq", "wiki/nope")),
            ("相对路径链接", doc.replace("wiki/prereq", "../prereq/index.html")),
            ("落单的 $", doc.replace("因为要自检。", "花了 $5。")),
            ("引用块", doc.replace("补充内容。", "> 引用")),
            ("代码块未结束", doc.replace("print(2 * 1)\n```", "print(2 * 1)")),
            ("自己写本章问题", doc.replace("## 边界", "## 本章问题")),
            ("折叠块前缀", doc.replace("::: 补充：更多", "::: 更多")),
            ("描述含公式", doc.replace("生成器自检页。", "自检 $x$。")),
        ]:
            md.write_text(broken, encoding="utf-8")
            try:
                build(md, wiki_root=wiki)
                checks.append((f"{name}应当报错", False))
            except SystemExit:
                checks.append((f"{name}应当报错", True))

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(("  OK  " if ok else "  FAIL") + " " + name)
    if failed:
        print(f"生成器自检失败：{len(failed)} 项")
        return 1
    print(f"生成器自检通过：{len(checks)} 项")
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="读 research/concept.md，生成页面目录下的 index.html 和 overview.html")
    ap.add_argument("md", nargs="?", type=Path, help="wiki/<name>/research/concept.md")
    ap.add_argument("-o", "--out", type=Path, help="输出目录，默认 wiki/<name>/")
    ap.add_argument("--run", action="store_true", help="运行文中的 Python 代码并核对输出")
    ap.add_argument("--selftest", action="store_true", help="用虚构数据自检生成器")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(_selftest())
    if a.md is None:
        ap.error("需要 concept.md 的路径")
    if not a.md.exists():
        raise SystemExit(f"文件不存在：{a.md}")
    index_path, overview_path, stats, ctx = build(a.md, a.out)
    print(f"已生成 {index_path} 和 {overview_path.name}：{stats['chapters']} 章，{stats['details']} 个折叠块")
    if a.run:
        ran, problems = run_code(ctx.fences)
        for p in problems:
            print(p)
        print(f"运行了 {ran} 段代码，{len(problems)} 段有问题")
        raise SystemExit(1 if problems else 0)
