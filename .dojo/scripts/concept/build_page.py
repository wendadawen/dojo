#!/usr/bin/env python3
"""读 research/content.md，生成长页 index.html 和短页 overview.html；格式见 guides/concept/content.md。"""

from __future__ import annotations

import html
import itertools
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NoReturn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import LONG_LIMIT, SHORT_LIMIT, read_text, report as report_checks  # noqa: E402

TEMPLATES = HERE.parents[1] / "templates" / "concept"
REPO_ROOT = HERE.parents[2]
CONTENT_MARKER = "<!-- @content -->"

META_KEYS = {"描述": "description", "摘要": "summary", "主题": "topics", "标签": "tag"}
SECTIONS = ("资料", "短页", "长页")

MATH_RE = re.compile(r"\$\$.+?\$\$|(?<!\\)\$(?!\$)(?:\\.|[^$\n])+?(?<!\\)\$", re.S)
LIST_RE = re.compile(r"^(\s*)(-|\d+\.)\s+(.*)$")
RAW_HTML_RE = re.compile(r"^<(figure|div|svg)\b")
NUMBERED_RE = re.compile(r"^\d+(?:\.\d+)*[.、．\s]")
CJK_RE = re.compile(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]")
HAN_RE = re.compile(r"[\u4e00-\u9fff]")
WORD_RE = re.compile(r"[A-Za-z]+")
DETAILS_RE = re.compile(r"<details\b(?:(?!<details\b)[\s\S])*?</details>")
BASIS_RE = re.compile(r"^依据[：:]\s*(.*)$")

Line = tuple[int, str]


def fail(no: int, msg: str) -> NoReturn:
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
    buf, j = [s], i
    if not (len(s) > 4 and s.endswith("$$")):
        while True:
            j += 1
            if j >= len(lines):
                fail(no, "独立公式缺少结尾的 $$")
            buf.append(lines[j][1].strip())
            if buf[-1].endswith("$$"):
                break
    out.append("<p>" + html.escape("\n".join(buf), quote=False) + "</p>")
    return j + 1


def _fold(lines: list[Line], i: int, ctx: Ctx, out: list[str]) -> int:
    no, raw = lines[i]
    title = raw.strip()[3:].strip()
    if not title:
        fail(no, "折叠块写成「::: <标题>」，标题写明里面是什么")
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
    only_code = all(b.startswith('<div class="code-block">') for b in inner)
    cls = ' class="code-details"' if only_code else ""
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
    """表格一行拆成单元格：公式里的 | 不拆，公式外的 \\| 是字面竖线，\\$ 是字面美元。"""
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
        if c == "\\" and s[k + 1:k + 2] == "$":
            buf += "\\$"
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
    # 先查落单的 $，再切列：不然后续的列数断言会报出误导性的「列数不一致」
    for no, r in rows:
        if "$" in MATH_RE.sub("", r.replace("\\$", "")):
            fail(no, f"有落单的 $：{r.strip()[:40]}")
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


def render_blocks(lines: list[Line], ctx: Ctx, heading=None, level: str = "") -> list[str]:
    """一段 Markdown 行转成 HTML 块。只允许 level 这一级标题，交给 heading 渲染。"""
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
            i = _fold(lines, i, ctx, out)
        elif RAW_HTML_RE.match(s):
            i = _raw_html(lines, i, out)
        elif s.startswith("#"):
            m = re.fullmatch(r"(#+)\s+(.+)", s)
            if heading is None or not m or m.group(1) != level:
                fail(no, f"这里只能写 {level} 标题" if level else "这里不能写标题")
            out.append(heading(m.group(2).strip(), no))
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
    """切分标题、页面信息、资料、短页、长页；「依据：」行从正文取出，basis 是「行号 → 依据」。"""
    title, meta = "", {}
    sections: dict[str, list[Line]] = {}
    current, fence = None, False
    for no, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if s.startswith("```"):
            fence = not fence
        if not fence and line.startswith("## "):
            name = line[3:].strip()
            if name not in SECTIONS:
                fail(no, f"只有「资料」「短页」「长页」三节，没有「{name}」")
            if name in sections:
                fail(no, f"「{name}」重复了")
            current = name
            sections[name] = []
        elif current is not None:
            sections[current].append((no, line))
        elif line.startswith("# "):
            if title:
                fail(no, "只能有一个 # 标题")
            title = line[2:].strip()
        elif s:
            m = re.match(r"-\s*([^：:]+)[：:]\s*(.*)", s)
            if not title or not m or m.group(1).strip() not in META_KEYS:
                fail(no, "开头先写「# 概念名」，下面只写 描述、摘要、主题、标签 四行")
            meta[META_KEYS[m.group(1).strip()]] = m.group(2).strip()

    if not title:
        fail(0, "缺少「# 概念名」")
    if "$" in title or "：" in title:
        fail(0, "# 后面只写概念名")
    for zh, key in META_KEYS.items():
        if not meta.get(key):
            fail(0, f"缺少页面信息：{zh}")
    if "$" in meta["description"]:
        fail(0, "描述是纯文本，公式写到摘要里")
    for name in SECTIONS:
        if not any(l.strip() for _, l in sections.get(name, [])):
            fail(0, f"缺少「{name}」一节")

    refs: list[dict] = []
    for no, line in sections["资料"]:
        s = line.strip()
        if not s:
            continue
        if not line.startswith("- "):
            fail(no, "资料每条写成顶格的「- 」开头的一行，只写出处，不写摘录")
        refs.append({"no": no, "text": s[2:].strip()})

    basis: dict[int, str] = {}
    for name in ("短页", "长页"):
        kept: list[Line] = []
        fence = False
        for no, line in sections[name]:
            if line.strip().startswith("```"):
                fence = not fence
            m = None if fence else BASIS_RE.match(line.strip())
            if m:
                if not m.group(1).strip():
                    fail(no, "「依据：」后面要写文件和位置，或写「自行构造」")
                basis[no] = m.group(1).strip()
            else:
                kept.append((no, line))
        sections[name] = kept

    opening, chapters = split_long(sections["长页"])
    if not any(l.strip() for _, l in opening):
        fail(sections["长页"][0][0], "长页在第一个 ### 之前要有开头")
    if not chapters:
        fail(sections["长页"][0][0], "长页至少有一章：### <章标题>")
    for no, name, _ in chapters:
        if NUMBERED_RE.match(name):
            fail(no, "章的标题不写编号，生成器会编号")

    return {"title": title, "meta": meta, "short": sections["短页"],
            "opening": opening, "chapters": chapters, "refs": refs, "basis": basis}


def split_long(lines: list[Line]) -> tuple[list[Line], list[tuple[int, str, list[Line]]]]:
    """长页按 ### 切成开头和各章。折叠块和代码块里的行不参与切分。"""
    opening: list[Line] = []
    chapters: list[tuple[int, str, list[Line]]] = []
    fence, depth = False, 0
    for no, line in lines:
        s = line.strip()
        if s.startswith("```"):
            fence = not fence
        elif not fence and s.startswith(":::"):
            depth += -1 if s == ":::" else 1
        elif not fence and re.match(r"###\s", s):
            if depth:
                fail(no, "折叠块里不能写标题")
            chapters.append((no, s[4:].strip(), []))
            continue
        (chapters[-1][2] if chapters else opening).append((no, line))
    return opening, chapters


def _open_counts(fragment: str) -> tuple[int, int]:
    """折叠块外的汉字数和英文单词数，一段公式算一个词。"""
    while DETAILS_RE.search(fragment):
        fragment = DETAILS_RE.sub(" ", fragment)
    text = MATH_RE.sub(" x ", html.unescape(re.sub(r"<[^>]+>", " ", fragment)))
    return len(HAN_RE.findall(text)), len(WORD_RE.findall(text))


def count_words(fragment: str) -> int:
    return sum(_open_counts(fragment))


def reading_minutes(fragment: str) -> int:
    han, words = _open_counts(fragment)
    return max(1, math.ceil(han / 350 + words / 200))


def render(doc: dict, ctx: Ctx) -> tuple[str, str, dict]:
    parts = render_blocks(doc["opening"], ctx)
    for k, (no, name, body) in enumerate(doc["chapters"], 1):
        parts.append(f'<h2 id="s{k}">{k}. {inline(name, no, ctx)}</h2>')
        counter = itertools.count(1)

        def sub(text: str, line_no: int, k: int = k, counter=counter) -> str:
            j = next(counter)
            return f'<h3 id="s{k}-{j}">{k}.{j} {inline(text, line_no, ctx)}</h3>'

        parts += render_blocks(body, ctx, heading=sub, level="####")
    reading = "\n\n".join(parts)
    parts.append('<h2 id="references">参考资料</h2>\n<ol>\n'
                 + "\n".join(f"<li>{inline(r['text'], r['no'], ctx)}</li>" for r in doc["refs"])
                 + "\n</ol>")
    short = render_blocks(doc["short"], ctx, level="###",
                          heading=lambda t, no: f"<h2>{inline(t, no, ctx)}</h2>")
    short_html = "\n\n".join(short)
    lead = html.escape(doc["meta"]["description"])
    stats = {"chapters": len(doc["chapters"]), "details": ctx.details,
             "long": count_words(reading), "minutes": reading_minutes(reading),
             "short": count_words(lead + " " + short_html)}
    return "\n\n".join(parts), short_html, stats


def _fill(tpl: str, out_dir: Path, pairs: list[tuple[str, str]]) -> str:
    target = out_dir.resolve()
    if REPO_ROOT in target.parents:
        up = os.path.relpath(REPO_ROOT, target).replace(os.sep, "/")
        tpl = tpl.replace('"../../', f'"{up}/')
    for old, _ in pairs:
        if old not in tpl:
            fail(0, f"模板里找不到「{old[:30]}」，模板改了？")
    leftover = set(re.findall(r"【[^】]*】", tpl)) - {old for old, _ in pairs}
    if leftover:
        fail(0, f"模板里有没处理的占位符：{sorted(leftover)}")
    return re.sub("|".join(re.escape(old) for old, _ in pairs), lambda m: dict(pairs)[m.group(0)], tpl)


def build(md: Path, wiki_root: Path | None = None) -> tuple[Path, Path, dict, Ctx]:
    """research/content.md → 上一级目录下的 index.html 和 overview.html。"""
    if md.parent.name != "research":
        fail(0, f"{md} 要放在 research/ 下")
    out_dir = md.parent.parent
    ctx = Ctx(out_dir, wiki_root or REPO_ROOT / "wiki")
    doc = parse(read_text(md))
    body, short, stats = render(doc, ctx)
    meta, title = doc["meta"], doc["title"]

    index_html = _fill((TEMPLATES / "index.html").read_text(encoding="utf-8"), out_dir, [
        (CONTENT_MARKER, body),
        ("【首页摘要】", html.escape(meta["description"])),
        ("【首页可渲染摘要；公式使用 $...$】", html.escape(meta["summary"])),
        ("【主题，逗号分隔】", html.escape(meta["topics"])),
        ("【内容标签】", html.escape(meta["tag"])),
        ("【正文字数】", f"{stats['long']:,}"),
        ("【阅读分钟】", str(stats["minutes"])),
        ("【概念名】", html.escape(title)),
    ])

    overview_html = _fill((TEMPLATES / "overview.html").read_text(encoding="utf-8"), out_dir, [
        (CONTENT_MARKER, short),
        ("【概念名】", html.escape(title)),
        ("【主题标签】", html.escape(meta["tag"])),
        ("【定位摘要：这个概念是什么、解决什么问题】", html.escape(meta["description"])),
    ])

    out_dir.mkdir(parents=True, exist_ok=True)
    index_path, overview_path = out_dir / "index.html", out_dir / "overview.html"
    index_path.write_text(index_html, encoding="utf-8")
    overview_path.write_text(overview_html, encoding="utf-8")
    return index_path, overview_path, stats, ctx


def run_code(fences: list[tuple[int, str, str]]) -> tuple[int, list[str]]:
    """运行后面紧跟 text 代码块的 python 代码，把 text 块当预期输出逐行比对。"""
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


def report(index_path: Path, overview_path: Path, stats: dict) -> str:
    long_note = "" if stats["long"] <= LONG_LIMIT else f"，超出上限 {LONG_LIMIT}"
    short_note = "" if stats["short"] <= SHORT_LIMIT else f"，超出上限 {SHORT_LIMIT}"
    return (f"已生成 {index_path} 和 {overview_path.name}：长页 {stats['chapters']} 章、"
            f"{stats['details']} 个折叠块，折叠块外 {stats['long']} 字{long_note}；"
            f"短页 {stats['short']} 字{short_note}")


# ---------- 自检 ----------

def selftest() -> int:
    """用一份与任何真实概念无关的数据跑一遍，确认生成器可用、该报错的地方会报错。"""
    doc = "\n".join([
        "# 玩具算子",
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
        "## 短页",
        "",
        "### 是什么",
        "",
        "玩具算子把输入乘 2，前置见[前置页](wiki/prereq)。",
        "依据：book.txt 第 1 节",
        "",
        "## 长页",
        "",
        "开头一段，",
        "换行接着写。",
        "依据：book.txt 第 1 节",
        "",
        "### 定义",
        "",
        "公式如下：",
        "$$",
        "y = 2x,\\qquad a<b",
        "$$",
        "",
        "- $y$：输出",
        "- 嵌套：",
        "  - 第二层 **加粗**",
        "依据：自行构造",
        "",
        "| 输入 | 输出 |",
        "|---|---:|",
        "| $1$ | $2$ |",
        "| $\\|x\\|$ | a\\|b |",
        "",
        "#### 小节",
        "",
        "::: 例子的完整过程",
        "```python",
        "print(2 * 1)",
        "```",
        "",
        "```text",
        "2",
        "```",
        ":::",
        "",
        "### 边界",
        "",
        "源码摘录不运行：",
        "",
        "```python",
        "def forward(self, x):",
        "    return self.missing(x)",
        "```",
        "",
        "::: 推导",
        "折叠块里的字不计入。",
        "::: 更深一层",
        "嵌套折叠。",
        ":::",
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
        md = pdir / "content.md"
        md.write_text(doc, encoding="utf-8")
        idx, ov, stats, ctx = build(md, wiki_root=wiki)
        t, o = idx.read_text(encoding="utf-8"), ov.read_text(encoding="utf-8")
        checks += [
            ("输出到页面目录", idx == root / "wiki" / "toy" / "index.html" and ov.exists()),
            ("无占位符与标记残留", "【" not in t + o and "@content" not in t + o),
            ("样式改为外链", "dojo-concept.css" in t and "<style>" not in t),
            ("标题与导航", "<h1 class=\"title\">玩具算子</h1>" in t
             and "<title>玩具算子</title>" in t
             and '<span class="nav-brand">玩具算子</span>' in t),
            ("开头在第一章之前", t.index("开头一段，换行接着写。") < t.index('<h2 id="s1">')),
            ("章编号", '<h2 id="s1">1. 定义</h2>' in t and '<h2 id="s2">2. 边界</h2>' in t),
            ("小节编号", '<h3 id="s1-1">1.1 小节</h3>' in t),
            ("只有代码的折叠块", '<details class="code-details">' in t and "language-python" in t),
            ("普通折叠块与嵌套", "<summary>推导</summary>" in t and "<summary>更深一层</summary>" in t),
            ("公式原样保留、< 转义", "y = 2x,\\qquad a&lt;b" in t),
            ("表格右对齐", 'style="text-align:right"' in t),
            ("表格里公式的竖线不拆", "$\\|x\\|$" in t and "a|b" in t),
            ("嵌套列表", t.count("<ul>") == 2 and "<b>加粗</b>" in t),
            ("参考资料", '<h2 id="references">参考资料</h2>' in t
             and 'href="https://example.com/paper"' in t),
            ("短页：小标题与站内链接", "<h2>是什么</h2>" in o and 'href="../prereq/index.html"' in o),
            ("短页：描述进 lead", '<p class="lead">生成器自检页。</p>' in o),
            ("短页内容不进长页", "玩具算子把输入乘 2" not in t),
            ("阅读时间写进长页", f"正文 {stats['long']:,} 字" in t),
            ("折叠块外字数不含折叠块", count_words("<p>一二</p><details><summary>三</summary>"
                                          "<details>四</details></details>") == 2),
            ("统计", stats["chapters"] == 2 and stats["details"] == 3),
            ("依据行不出现在页面里", "依据" not in t + o and "book.txt" not in t + o),
            ("依据行被取出", sorted(parse(doc)["basis"].values()).count("book.txt 第 1 节") == 2
             and "自行构造" in parse(doc)["basis"].values()),
        ]
        ran, problems = run_code(ctx.fences)
        checks.append(("--run：只跑带输出块的代码，输出一致", ran == 1 and not problems))
        ran, problems = run_code([(1, "python", "print(3)"), (2, "text", "2")])
        checks.append(("--run：输出不一致会报出", ran == 1 and len(problems) == 1))

        again, _, _, _ = build(md, wiki_root=wiki)
        checks.append(("幂等", again.read_text(encoding="utf-8") == t))

        md.write_text(doc.replace("嵌套折叠。", "【注】【概念名】"), encoding="utf-8")
        t_note = build(md, wiki_root=wiki)[0].read_text(encoding="utf-8")
        checks.append(("正文含【】也能生成", "【注】【概念名】" in t_note))
        md.write_text(doc, encoding="utf-8")

        for name, broken in [
            ("缺少短页", doc.replace("## 短页", "### 短页")),
            ("多出一节", doc.replace("## 长页", "## 其他\n\n文字\n\n## 长页")),
            ("长页没有开头", doc.replace("开头一段，\n换行接着写。\n", "")),
            ("长页没有章", doc.replace("### 定义", "定义").replace("### 边界", "边界")
             .replace("#### 小节", "小节")),
            ("章的标题写编号", doc.replace("### 定义", "### 1. 定义")),
            ("标题带一句话", doc.replace("# 玩具算子", "# 玩具算子：乘二")),
            ("缺少标签", doc.replace("- 标签：数学与数值\n", "")),
            ("开头写小节标题", doc.replace("开头一段，", "#### 开头小节\n\n开头一段，")),
            ("折叠块里写标题", doc.replace("折叠块里的字不计入。", "### 藏起来的章")),
            ("折叠块没有标题", doc.replace("::: 推导", ":::")),
            ("站内链接指向不存在的页面", doc.replace("wiki/prereq", "wiki/nope")),
            ("相对路径链接", doc.replace("wiki/prereq", "../prereq/index.html")),
            ("落单的 $", doc.replace("嵌套折叠。", "花了 $5。")),
            ("引用块", doc.replace("嵌套折叠。", "> 引用")),
            ("代码块未结束", doc.replace("print(2 * 1)\n```", "print(2 * 1)")),
            ("描述含公式", doc.replace("生成器自检页。", "自检 $x$。")),
            ("依据行后面是空的", doc.replace("依据：自行构造", "依据：")),
            ("资料下面缩进写摘录", doc.replace("- 虚构论文：[示例](https://example.com/paper)",
                                       "- 虚构论文：[示例](https://example.com/paper)\n  - 摘录：原文")),
        ]:
            md.write_text(broken, encoding="utf-8")
            try:
                build(md, wiki_root=wiki)
                checks.append((f"{name}应当报错", False))
            except SystemExit:
                checks.append((f"{name}应当报错", True))

    return report_checks(checks, "生成器")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("md", nargs="?", type=Path, help="wiki/<name>/research/content.md")
    ap.add_argument("--run", action="store_true", help="运行文中的 Python 代码并核对输出")
    ap.add_argument("--selftest", action="store_true", help="用虚构数据自检生成器")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    if a.md is None:
        ap.error("需要 content.md 的路径")
    if not a.md.exists():
        fail(0, f"文件不存在：{a.md}")
    index_path, overview_path, stats, ctx = build(a.md)
    print(report(index_path, overview_path, stats))
    import check_content  # 它反过来 import 本文件，放到这里避免循环
    problems = check_content.check(a.md)
    for p in problems:
        print(p)
    print(f"检查：{len(problems)} 个问题" if problems else "检查通过")
    failed = bool(problems)
    if a.run:
        ran, problems = run_code(ctx.fences)
        for p in problems:
            print(p)
        print(f"运行了 {ran} 段代码，{len(problems)} 段有问题")
        failed = failed or bool(problems)
    raise SystemExit(1 if failed else 0)
