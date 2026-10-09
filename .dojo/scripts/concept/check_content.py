#!/usr/bin/env python3
"""检查 content.md：依据、数字、结构、链接、字数、代码块。语义上的判断交给审查者。"""

from __future__ import annotations

import html
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "site"))
import build_page  # noqa: E402
import check_outline  # noqa: E402
import vocab  # noqa: E402
from common import (  # noqa: E402
    CONTENT_ROUNDS, LAST_CHAPTER, LONG_LIMIT, SELF_MADE, SHORT_LIMIT, SHORT_SECTIONS, SNAPSHOT,
    check_basis_list, final_check_recorded, report,
)

REPO_ROOT = HERE.parents[2]
NAME_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
URL_RE = re.compile(r"https?://", re.I)
LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)\s]+)\)")
ISBN_RE = re.compile(r"ISBN[\s:：-]*[0-9Xx-]{10,17}")
NUM_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
TEXT_SUFFIXES = {".tex", ".txt", ".md", ".bbl", ".rst", ".html", ".htm"}
# 接在句子后面的依据行：「依据：」后面是「自行构造」或「文件名.扩展名」；正文里自然出现的「判断依据：……」不算
BASIS_MID_RE = re.compile(r"\S.*依据[：:]\s*(?:自行构造|[\w./-]+(?:" + "|".join(re.escape(x) for x in TEXT_SUFFIXES) + r")\b)")
MAX_TEXT_BYTES = 2_000_000
BUDGET_SLACK = 1.3  # 一节折叠块外的字数，最多超出大纲预算这么多倍
SOURCE_WORDS = ("论文", "本文", "作者", "官方实现", "演示代码")  # 出处措辞，正文只写事实
REPO_DIR = "repo"  # guides/concept/collect.md 规定仓库 clone 到 sources/repo/


# ---------- 解析与小工具 ----------

class Content:
    """一份解析好的 content.md，各条规则共用。"""

    def __init__(self, md: Path, wiki_root: Path):
        text = md.read_text(encoding="utf-8")
        self.lines = text.splitlines()
        self.sources = md.parent / "sources"
        self.out_dir = md.parent.parent
        self.wiki_root = wiki_root
        self.doc = build_page.parse(text)  # 格式不对会抛 SystemExit，由 check() 接住
        self.basis = self.doc["basis"]
        self.parts = [self.doc["short"], self.doc["opening"]] + [b for _n, _t, b in self.doc["chapters"]]
        self.body = [ln for part in self.parts for ln in part]
        self._render = None
        self.body_html = ""

    def rendered(self):
        """渲染一次，得到字数统计和代码块。格式错误时抛 SystemExit。"""
        if self._render is None:
            ctx = build_page.Ctx(self.out_dir, self.wiki_root)
            _body, _short, stats = build_page.render(self.doc, ctx)
            self._render = (stats, ctx.fences)
            self.body_html = _body
        return self._render

    def node_words(self) -> dict[str, int]:
        """长页每个节点折叠块外的字数，键是节点标题，开头的键是「开头」。"""
        self.rendered()
        pieces = re.split(r"(?=<h[23] id=)", self.body_html.split('<h2 id="references">')[0])
        words = {}
        for piece in pieces:
            m = re.match(r'<h([23]) id="[^"]*">[\d.]+ (.*?)</h\1>', piece)
            title = html.unescape(re.sub(r"<[^>]+>", "", m.group(2))) if m else "开头"
            words[title] = words.get(title, 0) + build_page.count_words(piece)
        return words


def outside_fences(lines):
    """逐行产出代码块之外的行。"""
    fence = False
    for no, line in lines:
        if line.strip().startswith("```"):
            fence = not fence
            continue
        if not fence:
            yield no, line


def without_math(line: str) -> str:
    return build_page.MATH_RE.sub(" ", line)


def text_files(sources: Path) -> list[Path]:
    """sources/ 下用来核对数字的文本：递归读论文源码、文本快照，跳过 repo/、PDF、图片和过大的文件。"""
    if not sources.is_dir():
        return []
    out = []
    for p in sorted(sources.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if p.relative_to(sources).parts[0] == REPO_DIR or p.stat().st_size > MAX_TEXT_BYTES:
            continue
        out.append(p)
    return out


def paragraphs(lines):
    """把一段行切成文字段落：代码块、折叠块、标题、表格、公式行把段落隔开。每段是 [(行号, 行)]。"""
    para, fence = [], False
    for no, line in lines:
        s = line.strip()
        if s.startswith("```"):
            fence = not fence
            if para:
                yield para
                para = []
            continue
        if fence or s.startswith(("|", ":::", "#", "$$")) or not s:
            if para:
                yield para
                para = []
        else:
            para.append((no, line))
    if para:
        yield para


# ---------- 规则：每条一个函数，返回问题列表 ----------

def dir_name_is_kebab(c: Content) -> list[str]:
    if NAME_RE.fullmatch(c.out_dir.name):
        return []
    return [f"目录名：{c.out_dir.name} 要用小写英文、数字和连字符"]


def topics_and_tag_in_vocab(c: Content) -> list[str]:
    errors, meta = [], c.doc["meta"]
    for topic in [t.strip() for t in re.split(r"[,，]", meta["topics"]) if t.strip()]:
        if topic not in vocab.ALLOWED_TOPICS:
            errors.append(f"主题：「{topic}」不在词表里（可选：{'、'.join(vocab.ALLOWED_TOPICS)}）")
    if re.search(r"[,，]", meta["tag"]):
        errors.append(f"标签：只能写一个，现在是「{meta['tag']}」")
    elif meta["tag"] not in vocab.ALLOWED_TAGS:
        errors.append(f"标签：「{meta['tag']}」不在词表里（可选：{'、'.join(vocab.ALLOWED_TAGS)}）")
    return errors


def references_have_link(c: Content) -> list[str]:
    return [f"第 {r['no']} 行：资料要带链接，没有网址的教材写 ISBN（guides/concept/content.md「格式」）"
            for r in c.doc["refs"] if not (URL_RE.search(r["text"]) or ISBN_RE.search(r["text"]))]


def short_has_three_sections(c: Content) -> list[str]:
    # 标题文字不查，合不合适交给审大纲
    n = sum(1 for _no, line in c.doc["short"] if line.strip().startswith("### "))
    if n == SHORT_SECTIONS:
        return []
    return [f"短页：要恰好 {SHORT_SECTIONS} 节，现在是 {n} 节（guides/concept.md「短页」）"]


def short_has_no_fold_code_table(c: Content) -> list[str]:
    errors = []
    for no, line in c.doc["short"]:
        s = line.strip()
        for mark, what in (":::", "折叠块"), ("```", "代码块"), ("|", "表格"):
            if s.startswith(mark):
                errors.append(f"第 {no} 行：短页不放{what}（guides/concept.md「短页」）")
    return errors


def long_ends_with_boundary(c: Content) -> list[str]:
    chapters = c.doc["chapters"]
    errors = []
    if chapters[-1][1] != LAST_CHAPTER:
        errors.append(f"第 {chapters[-1][0]} 行：长页最后一章的标题写「{LAST_CHAPTER}」，"
                      f"现在是「{chapters[-1][1]}」（guides/concept.md「长页」）")
    if len(chapters) < 2:
        errors.append(f"长页：除「{LAST_CHAPTER}」外，至少还要有一章讲机制")
    return errors


def no_external_url_in_body(c: Content) -> list[str]:
    return [f"第 {no} 行：正文里不写外部网址，出处放进「资料」（guides/concept/content.md「格式」）"
            for no, line in outside_fences(c.body) if URL_RE.search(without_math(line))]


def links_point_to_existing_pages(c: Content) -> list[str]:
    errors = []
    for no, line in outside_fences(c.body):
        for href in LINK_RE.findall(without_math(line)):
            if href.startswith(("assets/", "#")):
                continue
            m = re.fullmatch(r"wiki/([\w.-]+)(?:/(?:index|overview)\.html)?(?:#.*)?", href)
            if not m:
                errors.append(f"第 {no} 行：链接 {href} 只能写 wiki/<页面名>（guides/concept/content.md「格式」）")
            elif not (c.wiki_root / m.group(1) / "index.html").exists():
                errors.append(f"第 {no} 行：链接的页面 wiki/{m.group(1)} 不存在，先按指南生成它")
    return errors


def every_paragraph_has_basis(c: Content) -> list[str]:
    """依据行已被 parse 取走、只剩行号；同一区块里引导句加后面的列表，共用区块末尾的一行依据。"""
    if not c.basis:
        return ["依据：全文没有一行「依据：…」，每段正文后面都要写（guides/concept/content.md「怎么写」）"]
    errors, basis = [], sorted(c.basis)
    for lines in c.parts:
        fence = False
        pending = None  # 当前区块里第一段还没遇到依据行的文字
        last = 0
        for no, line in lines + [(10**9, "###")]:  # 末尾补一个区块边界
            s = line.strip()
            if pending is not None and any(last < n < no for n in basis):
                pending = None
            last = no
            if s.startswith("```"):
                fence = not fence
                continue
            if fence or not s or s.startswith(("$$", "|", "!", "<")):
                continue
            if s.startswith(("#", ":::")):
                if pending is not None:
                    errors.append(f"第 {pending} 行：这段正文后面没有「依据：」行（guides/concept/content.md「怎么写」）")
                pending = None
            elif pending is None:
                pending = no
    return errors


def basis_matches_sources(c: Content) -> list[str]:
    errors = []
    for no, text in c.basis.items():
        for why in check_basis_list(text, c.sources):
            errors.append(f"第 {no} 行：依据对不上：{why}")
    return errors


def basis_on_its_own_line(c: Content) -> list[str]:
    # 接在句子后面的「依据：」不会被生成器取出，会留在页面上
    return [f"第 {no} 行：「依据：」要单独写成一行，放在这段正文的下一行，"
            "不能接在句子后面，否则它会出现在页面上（guides/concept/content.md「怎么写」）"
            for no, line in outside_fences(c.body) if BASIS_MID_RE.search(line)]


def numbers_appear_in_sources(c: Content) -> list[str]:
    """页面里出现的数字，必须在 sources 的快照里作为一个完整的数出现过，或所在段落标了自行构造。
    公式和代码块里的数字不查。"""
    files = text_files(c.sources)
    if not files:
        return ["数字：research/sources/ 下没有快照，无法核对页面里的数字"]
    hay = " ".join(p.read_text(encoding="utf-8", errors="ignore") for p in files).replace(",", "")
    known = set(NUM_RE.findall(hay))
    errors, basis = [], c.basis
    for lines in c.parts:
        for para in paragraphs(lines):
            last = para[-1][0]
            nxt = min((n for n in basis if n > last), default=None)  # 依据行在这段最后一行之后
            if nxt is not None and basis[nxt].strip() == SELF_MADE and nxt - last <= 2:
                continue
            for no, line in para:
                probe = re.sub(r"`[^`]*`|\[[^\]]*\]\([^)]*\)", " ", without_math(line))
                for n in NUM_RE.findall(probe.replace(",", "")):
                    if n not in known:
                        errors.append(f"第 {no} 行：数字 {n} 在 research/sources/ 里找不到；"
                                      "资料里没有的数字，要么删掉，要么改成自己构造的例子并写「依据：自行构造」")
    return errors


def words_within_limit(c: Content) -> list[str]:
    stats, _ = c.rendered()
    errors = []
    if stats["long"] > LONG_LIMIT:
        errors.append(f"字数：长页折叠块外 {stats['long']} 字，超出上限 {LONG_LIMIT}；"
                      "把推导、其余公式、例子过程、实验条件收进折叠块，或删掉（guides/concept.md「长页」）")
    if stats["short"] > SHORT_LIMIT:
        errors.append(f"字数：短页 {stats['short']} 字，超出上限 {SHORT_LIMIT}（guides/concept.md「短页」）")
    return errors


def code_blocks_are_python_with_output(c: Content) -> list[str]:
    _, fences = c.rendered()
    errors = []

    def adjacent(fence, next_no):
        """fence 的结尾和下一个代码块的开头之间只有空行。"""
        no, _lang, code = fence
        end = no + code.count("\n") + 2 if code else no + 1
        return all(not ln.strip() for ln in c.lines[end:next_no - 1])

    for k, (no, lang, code) in enumerate(fences):
        if lang == "text":
            prev = fences[k - 1] if k else None
            if not prev or prev[1] != "python" or not adjacent(prev, no):
                errors.append(f"第 {no} 行：text 代码块只用来写 python 代码的输出，要紧跟在它后面"
                              "（guides/concept/content.md「格式」）")
        elif lang == "python":
            nxt = fences[k + 1] if k + 1 < len(fences) else None
            if not nxt or nxt[1] != "text" or not adjacent((no, lang, code), nxt[0]):
                errors.append(f"第 {no} 行：python 代码后面要紧跟一个写运行输出的 text 代码块，"
                              "否则无法核对（guides/concept/content.md「格式」）")
        else:
            errors.append(f"第 {no} 行：代码块只能是 python 加输出块，现在是 {lang}；"
                          "换成文字或表格来写例子（guides/concept/content.md「格式」）")
    return errors


def no_source_wording(c: Content) -> list[str]:
    errors = []
    for no, line in outside_fences(c.body):
        for w in SOURCE_WORDS:
            if w in re.sub(r"`[^`]*`", " ", without_math(line)):
                errors.append(f"第 {no} 行：正文不写「{w}」这类出处措辞，直接写事实，出处放进「资料」"
                              "（guides/concept/content.md「怎么写」）")
    for key in ("description", "summary"):
        errors += [f"{key}：不写「{w}」这类出处措辞" for w in SOURCE_WORDS if w in c.doc["meta"][key]]
    return errors


def references_are_in_sources_index(c: Content) -> list[str]:
    index = c.sources / "index.md"
    urls = [(r["no"], u) for r in c.doc["refs"] for u in dict.fromkeys(re.findall(r"https?://[^\s)\]]+", r["text"]))]
    if urls and not index.is_file():
        return ["资料：research/sources/index.md 不存在，「资料」里的链接都要先收进去（guides/concept/collect.md）"]
    text = index.read_text(encoding="utf-8") if urls else ""
    return [f"第 {no} 行：资料 {u} 没有收进 research/sources/index.md，只列收了的资料（guides/concept/collect.md）"
            for no, u in urls if u not in text]


def description_is_one_sentence(c: Content) -> list[str]:
    if c.doc["meta"]["description"].strip().rstrip("。").count("。"):
        return ["描述：只写一句话，第二句挪到摘要（guides/concept/outline.md「格式」）"]
    return []


def sections_within_budget(c: Content) -> list[str]:
    outline = c.sources.parent / "outline.md"
    if not outline.is_file():
        return ["大纲：research/outline.md 不存在，正文要按审过的大纲写"]
    budget = {t.lstrip("# ").strip(): int(w) for _no, t, _what, w, _b in check_outline.parse(outline)["nodes"]["长页"]
              if w.isdigit()}
    actual = c.node_words()
    errors = [f"大纲里的「{t}」在正文里找不到，节点要一一对应，要改回去改大纲（guides/concept/content.md「怎么写」）"
              for t in budget if t not in actual]
    errors += [f"正文里的「{t}」大纲里没有，节点要一一对应，要改回去改大纲" for t in actual if t not in budget]
    errors += [f"「{t}」折叠块外 {actual[t]} 字，大纲预算 {budget[t]}，超出 {BUDGET_SLACK} 倍；"
               "删重复，或把推导、数字收进折叠块，不要回头改预算（guides/concept/content.md「怎么写」）"
               for t in budget if t in actual and budget[t] and actual[t] > budget[t] * BUDGET_SLACK]
    return errors


def long_has_self_made_example(c: Content) -> list[str]:
    lines = {no for _n, _t, body in c.doc["chapters"] for no, _l in body} | {no for no, _l in c.doc["opening"]}
    if any(basis.strip() == SELF_MADE and no - 1 in lines for no, basis in c.basis.items()):
        return []
    return [f"长页：至少要有一段依据写「{SELF_MADE}」的例子，小数字，能跟着走完（guides/concept.md「长页」）"]


def folds_promised_by_outline_exist(c: Content) -> list[str]:
    """大纲「写什么」里写了「折叠」的长页节点，正文那一节里要真有折叠块。"""
    outline = c.sources.parent / "outline.md"
    if not outline.is_file():
        return []
    promised = [t.lstrip("# ").strip() for _no, t, what, *_ in check_outline.parse(outline)["nodes"]["长页"] if "折叠" in what]
    opening = [ln for _no, ln in c.doc["opening"]]
    bodies = {"开头": opening} | {name: [ln for _n, ln in body] for _no, name, body in c.doc["chapters"]}
    errors = []
    for title in promised:
        lines = bodies.get(title)
        if lines is None:
            continue  # 节点对不上由 sections_within_budget 报
        has_fold = any(ln.strip().startswith("::: ") for ln in lines)
        if not has_fold and not any(ln.strip().startswith("####") for ln in lines):
            errors.append(f"「{title}」：大纲里写了要折叠，正文这一节里没有折叠块（guides/concept/outline.md「写什么」）")
    return errors


def final_check_is_recorded(c: Content) -> list[str]:
    return final_check_recorded(c.sources.parent, "content", CONTENT_ROUNDS)


RULES = [
    dir_name_is_kebab, topics_and_tag_in_vocab, references_have_link,
    short_has_three_sections, short_has_no_fold_code_table, long_ends_with_boundary,
    no_external_url_in_body, links_point_to_existing_pages,
    every_paragraph_has_basis, basis_matches_sources, basis_on_its_own_line,
    numbers_appear_in_sources, words_within_limit, code_blocks_are_python_with_output,
    no_source_wording, references_are_in_sources_index, description_is_one_sentence,
    sections_within_budget, long_has_self_made_example, folds_promised_by_outline_exist, final_check_is_recorded,
]


def check(md: Path, wiki_root: Path | None = None) -> list[str]:
    """返回全部问题，每条一行。"""
    try:
        c = Content(md, wiki_root or REPO_ROOT / "wiki")
    except SystemExit as e:
        return [f"格式：{e}"]
    errors: list[str] = []
    for rule in RULES:
        try:
            errors += rule(c)
        except SystemExit as e:  # 渲染时发现的格式错误
            return errors + [f"格式：{e}"]
    return errors


# ---------- 自检 ----------

GOOD = "\n".join([
    "# 玩具算子", "",
    "- 描述：自检页。", "- 摘要：自检，与任何真实概念无关。",
    "- 主题：其他领域", "- 标签：通用概念", "",
    "## 资料", "",
    "- 虚构教材，第 1 版，[示例](https://example.com/book)", "",
    "## 短页", "",
    "### 它是什么", "", "玩具算子把输入乘 2。", "依据：book.md 第 1 章", "",
    "### 为什么要它", "", "用来演示。", "依据：自行构造", "",
    "### 怎么用它", "", "读入一个数，输出它的两倍。", "依据：book.md L3-L4", "",
    "## 长页", "", "开头一段。", "依据：book.md 第 1 章", "",
    "### 定义", "",
    "输出是输入的两倍，输入取 3.5 时输出 7.0，前置见[前置页](wiki/prereq)。", "依据：book.md 第 1 章", "",
    "```python", "print(2 * 1)", "```", "", "```text", "2", "```", "依据：自行构造", "",
    "### 适用条件与边界", "", "只对数值输入成立。", "依据：book.md 第 1 章",
])

# 与 GOOD 配套的大纲和资料目录；预算按 GOOD 各节的实际字数留足
OUTLINE = "\n".join([
    "# 大纲", "", "## 长页", "",
    "- 开头｜开头一段｜50｜book.md 第 1 章",
    "- ### 定义｜输出是输入的两倍｜100｜book.md 第 1 章",
    "- ### 适用条件与边界｜只对数值输入成立｜50｜book.md 第 1 章", "",
    "## 短页", "",
    "- ### 它是什么｜定义｜100｜book.md 第 1 章",
    "- ### 为什么要它｜动机｜100｜自行构造",
    "- ### 怎么用它｜用法｜100｜book.md L3-L4",
])
INDEX = "- https://example.com/book｜第 1 版\n"

# 每条规则至少一个坏输入，应当报出 want；want 为 None 的是不该报错的写法
CASES = [
    ("通过的页面没有报错", GOOD, None),
    ("资料没有链接", GOOD.replace("，[示例](https://example.com/book)", ""), "资料要带链接"),
    ("短页只有两节", GOOD.replace("### 怎么用它\n\n读入一个数，输出它的两倍。\n依据：book.md L3-L4\n\n", ""), "要恰好 3 节"),
    ("短页有四节", GOOD.replace("## 长页", "### 多出的一节\n\n内容。\n依据：自行构造\n\n## 长页", 1), "要恰好 3 节"),
    ("短页标题随便取也不报错", GOOD.replace("它是什么", "完全不同的标题"), None),
    ("短页放折叠块", GOOD.replace("读入一个数，输出它的两倍。", "读入一个数。\n\n::: 详情\n内容\n:::"), "短页不放折叠块"),
    ("短页放表格", GOOD.replace("读入一个数，输出它的两倍。", "|a|b|\n|---|---|\n|1|2|"), "短页不放表格"),
    ("最后一章标题不对", GOOD.replace("### 适用条件与边界", "### 什么时候不成立"), "最后一章"),
    ("正文写外部网址", GOOD.replace("输出是输入的两倍", "输出是输入的两倍（https://a.com）"), "不写外部网址"),
    ("链接的页面不存在", GOOD.replace("wiki/prereq", "wiki/nope"), "格式"),
    ("python 后面没有输出块", GOOD.replace("```text\n2\n```\n", ""), "要紧跟"),
    ("其他语言的代码块", GOOD.replace("```python\nprint(2 * 1)\n```\n\n```text\n2\n```", "```c\nint x;\n```"), "只能是 python"),
    ("长页超字数", GOOD.replace("开头一段。", "字" * (LONG_LIMIT + 1)), "长页折叠块外"),
    ("短页超字数", GOOD.replace("用来演示。", "字" * (SHORT_LIMIT + 1)), "短页"),
    ("依据的文件不存在", GOOD.replace("依据：book.md 第 1 章", "依据：nope.md 第 1 章", 1), "不在 research/sources/ 下"),
    ("依据的章节找不到", GOOD.replace("依据：book.md 第 1 章", "依据：book.md 第 9 章", 1), "里没有标题含"),
    ("依据的行号越界", GOOD.replace("book.md L3-L4", "book.md L3-L99"), "只有"),
    ("依据写成标题的一段文字", GOOD.replace("book.md 第 1 章", "book.md 玩具算子", 1), None),
    ("一段写两个依据", GOOD.replace("依据：book.md 第 1 章", "依据：book.md 第 1 章；book.md 1.2 另一节", 1), None),
    ("整页没有依据", "\n".join(l for l in GOOD.split("\n") if not l.startswith("依据：")), "全文没有一行"),
    ("依据接在句子后面", GOOD.replace("玩具算子把输入乘 2。\n依据：book.md 第 1 章", "玩具算子把输入乘 2。依据：book.md 第 1 章"), "要单独写成一行"),
    ("长页里依据接在句子后面", GOOD.replace("只对数值输入成立。\n依据：book.md 第 1 章", "只对数值输入成立。依据：book.md 第 1 章"), "要单独写成一行"),
    ("正文里自然出现的「依据：」不报", GOOD.replace("只对数值输入成立。", "判断的依据：资料里写得很清楚。"), None),
    ("代码块里的依据不查", GOOD.replace("print(2 * 1)", "print(2 * 1)  # 依据：无"), None),
    ("数字在资料里找不到", GOOD.replace("取 3.5 时输出 7.0", "取 3.5 时输出 8.0"), "数字 8.0"),
    ("自行构造的段落数字不查", GOOD.replace("用来演示。", "取 4242 试一下。"), None),
    ("数字在公式里不查", GOOD.replace("开头一段。", "开头一段，$x=9999$。"), None),
    ("短页有一段没写依据", GOOD.replace("用来演示。\n依据：自行构造\n", "用来演示。\n"), "后面没有「依据：」行"),
    ("长页有一段没写依据", GOOD.replace("只对数值输入成立。\n依据：book.md 第 1 章", "只对数值输入成立。"), "后面没有「依据：」行"),
    ("引导句和后面的列表共用一行依据", GOOD.replace("只对数值输入成立。", "分两步：\n\n1. 读入。\n2. 输出。"), None),
    ("正文写出处措辞", GOOD.replace("玩具算子把输入乘 2。", "论文说玩具算子把输入乘 2。"), "出处措辞"),
    ("描述写了两句", GOOD.replace("描述：自检页。", "描述：自检页。第二句。"), "只写一句话"),
    ("资料没有收进 index.md", GOOD.replace("https://example.com/book", "https://example.com/other"), "没有收进"),
    ("长页没有自行构造的例子", GOOD.replace("```text\n2\n```\n依据：自行构造", "```text\n2\n```\n依据：book.md 第 1 章"), "至少要有一段依据写"),
    ("小整数不能靠子串命中", GOOD.replace("开头一段。", "开头一段，共有 7 个。"), "数字 7"),
]


def selftest() -> int:
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        wiki = Path(tmp) / "wiki"
        (wiki / "prereq").mkdir(parents=True)
        (wiki / "prereq" / "index.html").write_text("<html></html>", encoding="utf-8")

        def make(name: str, text: str, with_sources: bool = True) -> Path:
            research = wiki / name / "research"
            research.mkdir(parents=True, exist_ok=True)
            if with_sources:
                (research / "sources").mkdir(exist_ok=True)
                (research / "sources" / "book.md").write_text(SNAPSHOT, encoding="utf-8")
                (research / "sources" / "index.md").write_text(INDEX, encoding="utf-8")
            (research / "outline.md").write_text(OUTLINE, encoding="utf-8")
            (research / "content.md").write_text(text, encoding="utf-8")
            return research / "content.md"

        md = make("toy", GOOD)
        for name, text, want in CASES:
            md.write_text(text, encoding="utf-8")
            errs = check(md, wiki)
            ok = (not errs) if want is None else any(want in e for e in errs)
            if want is None and errs:
                print("    意外报错：", errs)
            results.append((name, ok))
        outline = md.parent / "outline.md"
        md.write_text(GOOD.replace("开头一段。", "开头一段" + "字" * 80 + "。"), encoding="utf-8")
        results.append(("一节超出预算的 1.3 倍", any("大纲预算 50" in e for e in check(md, wiki))))
        fold = "\n\n::: 细节\n" + "字" * 500 + "\n依据：book.md 第 1 章\n:::\n"
        md.write_text(GOOD.replace("依据：book.md 第 1 章\n\n```python", "依据：book.md 第 1 章" + fold + "\n```python"), encoding="utf-8")
        errs = check(md, wiki)
        results.append(("折叠块里的字不算进预算", "大纲预算" not in "".join(errs)))
        md.write_text(GOOD, encoding="utf-8")
        outline.write_text(OUTLINE.replace("### 定义", "### 定义改了"), encoding="utf-8")
        results.append(("大纲节点和正文对不上", sum("一一对应" in e for e in check(md, wiki)) == 2))
        outline.write_text(OUTLINE, encoding="utf-8")
        outline.write_text(OUTLINE.replace("定义｜输出是输入的两倍", "定义｜输出是输入的两倍，推导折叠"), encoding="utf-8")
        results.append(("大纲写了折叠而正文没有", any("没有折叠块" in e for e in check(md, wiki))))
        md.write_text(GOOD.replace("```python\nprint(2 * 1)", "::: 推导\n两倍。\n依据：book.md 第 1 章\n:::\n\n```python\nprint(2 * 1)"), encoding="utf-8")
        results.append(("正文有折叠块不报", not any("没有折叠块" in e for e in check(md, wiki))))
        md.write_text(GOOD, encoding="utf-8")
        outline.write_text(OUTLINE, encoding="utf-8")
        record = md.parent / f"content-review-{CONTENT_ROUNDS}.md"
        record.write_text("## 处理\n全部采纳。\n", encoding="utf-8")
        results.append(("审到轮数上限没写最后一批改动核对", any("最后一批改动核对" in e for e in check(md, wiki))))
        record.write_text("## 最后一批改动核对\n无改动。\n", encoding="utf-8")
        results.append(("写了最后一批改动核对不报", not check(md, wiki)))
        results.append(("目录名不合规应当报错", any("目录名" in e for e in check(make("Bad_Name", GOOD), wiki))))
        results.append(("没有 sources 时报错", any("没有快照" in e for e in check(make("nosrc", GOOD, False), wiki))))
    return report(results, "正文检查")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    if len(sys.argv) != 2 or not Path(sys.argv[1]).exists():
        raise SystemExit("用法：check_content.py wiki/<name>/research/content.md")
    problems = check(Path(sys.argv[1]))
    for p in problems:
        print(p)
    print(f"检查：{len(problems)} 个问题" if problems else "检查通过")
    raise SystemExit(1 if problems else 0)
