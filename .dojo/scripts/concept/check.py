#!/usr/bin/env python3
"""概念页的机械检查：只查结构、字数、链接、代码块、大纲格式与「依据」是否核得上。

用法（从仓库根运行）：
    python3 .dojo/scripts/concept/check.py wiki/<name>/research/outline.md --outline
    python3 .dojo/scripts/concept/check.py wiki/<name>/research/content.md
    python3 .dojo/scripts/concept/check.py --selftest

规则全部报错，每条写明位置和原因，一次报完。语义上的判断（事实、范围、重复、文风、读不读得通）
不在这里，交给 guides/concept/ 里的审查者。build.py 生成完页面后会自动运行本脚本。

「依据」怎么核：
    依据：<文件名> <位置>[；<文件名> <位置>]
    文件名是 research/sources/ 下的文件。位置有两种写法：
      - 章节标题里的一段文字，如「2.3 Context-aware」，文件里要有一个标题行含这段文字；
      - 行号范围，如「L120-L180」，不能超出文件行数。
    「依据：自行构造」不核，只用于例子和推导，由复算审查者验证。
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]


sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "site"))
import build  # noqa: E402
import vocab  # noqa: E402

SHORT_SECTIONS = 3
LAST_CHAPTER = "适用条件与边界"
NAME_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
URL_RE = re.compile(r"https?://", re.I)
LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)\s]+)\)")
ISBN_RE = re.compile(r"ISBN[\s:：-]*[0-9Xx-]{10,17}")
LINES_RE = re.compile(r"[Ll](\d+)(?:\s*[-–~]\s*[Ll]?(\d+))?")
NUM_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
NODE_TYPES = ("正文", "公式", "例子", "折叠块")
SELF_MADE = "自行构造"
# 只认真正的依据行写法：「依据：」后面是「自行构造」，或「文件名.扩展名 位置」。正文里自然出现的「判断依据：……」不算
BASIS_MID_RE = re.compile(r"\S.*依据[：:]\s*(?:自行构造|[\w./-]+\.(?:tex|md|txt|html|pdf)\b)")


def outside_fences(lines: list[tuple[int, str]]):
    """逐行产出代码块之外的行。"""
    fence = False
    for no, line in lines:
        if line.strip().startswith("```"):
            fence = not fence
            continue
        if not fence:
            yield no, line


def body_lines(doc: dict) -> list[tuple[int, str]]:
    """短页和长页的全部行（含折叠块），不含资料。"""
    out = list(doc["short"]) + list(doc["opening"])
    for _no, _name, body in doc["chapters"]:
        out += body
    return out


def page_info(wiki_root: Path, page: str) -> tuple[str, str]:
    """读 wiki/<page>/index.html 的 dojo:type 和 dojo:status。"""
    text = (wiki_root / page / "index.html").read_text(encoding="utf-8")

    def meta(key: str) -> str:
        m = re.search(rf'<meta name="{key}" content="([^"]*)"', text)
        return m.group(1) if m else ""

    return meta("dojo:type"), meta("dojo:status") or "current"


# ---------- 依据 ----------

def _headings(path: Path) -> list[str]:
    """快照里的标题行：Markdown 的 # 行，或 LaTeX 的 \\section 一类。"""
    out = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        m = re.match(r"#{1,6}\s+(.*)", s) or re.match(r"\\(?:sub)*(?:section|paragraph)\*?\{(.*)\}", s)
        if m:
            out.append(m.group(1).strip())
    return out


def check_basis_item(item: str, sources: Path, cache: dict) -> str | None:
    """核一条「文件名 位置」，有问题返回原因，没问题返回 None。"""
    item = item.strip()
    m = re.match(r"(\S+)\s+(.+)", item)
    if not m:
        return f"「{item}」要写成「文件名 位置」"
    fname, where = m.group(1), m.group(2).strip()
    f = sources / fname
    if not f.is_file():
        return f"{fname} 不在 research/sources/ 下"
    lm = LINES_RE.fullmatch(where)
    if lm:
        total = len(f.read_text(encoding="utf-8", errors="ignore").splitlines())
        a = int(lm.group(1))
        b = int(lm.group(2) or lm.group(1))
        if a < 1 or b < a or b > total:
            return f"{fname} 只有 {total} 行，位置 {where} 不对"
        return None
    if f not in cache:
        cache[f] = _headings(f)
    if not any(where.lower() in h.lower() for h in cache[f]):
        return f"{fname} 里没有标题含「{where}」；章节写标题里的一段文字，或写行号范围如 L120-L180"
    return None


def _paragraphs_without_basis(doc: dict) -> list[str]:
    """每段正文后面都要有依据行。

    依据行已被 parse 从正文里取走，只留下行号。一段文字之后出现过依据行，这段就有依据；
    同一区块里一句引导加后面的有序列表，共用区块末尾的那一行。区块由标题、折叠块的
    开头结尾隔开。只管文字：代码块、公式、表格、图片不算段落。
    """
    errors: list[str] = []
    basis = sorted(doc["basis"])
    parts = [doc["short"], doc["opening"]] + [body for _no, _name, body in doc["chapters"]]
    for lines in parts:
        fence = False
        pending: int | None = None  # 当前区块里第一段还没遇到依据行的文字
        last = 0                    # 上一个处理过的行号
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
                    errors.append(_NO_BASIS.format(pending))
                pending = None
            elif pending is None:
                pending = no
    return errors


_NO_BASIS = "第 {} 行：这段正文后面没有「依据：」行（guides/concept/content.md「怎么写」）"


def check_basis(doc: dict, research: Path) -> list[str]:
    errors: list[str] = []
    sources = research / "sources"
    cache: dict = {}
    errors += _paragraphs_without_basis(doc)
    for no, text in doc["basis"].items():
        if text.strip() == SELF_MADE:
            continue
        for item in re.split(r"[;；]", text):
            if not item.strip():
                continue
            why = check_basis_item(item, sources, cache)
            if why:
                errors.append(f"第 {no} 行：依据对不上：{why}")
    if not doc["basis"]:
        errors.append("依据：全文没有一行「依据：…」，每段正文后面都要写（guides/concept.md「正文」）")
    return errors


TEXT_SUFFIXES = {".tex", ".txt", ".md", ".bbl", ".rst", ".html", ".htm"}
MAX_TEXT_BYTES = 2_000_000


def _text_files(sources: Path) -> list[Path]:
    """sources/ 下用来核对数字的文本：递归读，只读论文源码、文本快照这类，
    跳过仓库（代码和二进制太多）、PDF、图片和过大的文件。"""
    if not sources.is_dir():
        return []
    out = []
    for p in sorted(sources.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        rel = p.relative_to(sources).parts
        if rel[0] == "repo" or p.stat().st_size > MAX_TEXT_BYTES:
            continue
        out.append(p)
    return out


def check_numbers(doc: dict, research: Path) -> list[str]:
    """页面里出现的数字，必须在 research/sources/ 的快照里作为一个完整的数出现过，
    或所在段落标了自行构造。

    只在正文里查，公式和代码块里的数字不查，段落依据是「自行构造」的整段不查。
    """
    sources = research / "sources"
    files = _text_files(sources)
    if not files:
        return ["数字：research/sources/ 下没有快照，无法核对页面里的数字"]
    hay = " ".join(p.read_text(encoding="utf-8", errors="ignore") for p in files).replace(",", "")
    hay_numbers = set(NUM_RE.findall(hay))
    errors: list[str] = []
    for name in ("short", "opening"):
        errors += _numbers_in(doc[name], hay_numbers, doc["basis"])
    for _no, _name, body in doc["chapters"]:
        errors += _numbers_in(body, hay_numbers, doc["basis"])
    return errors


def _numbers_in(lines: list[tuple[int, str]], hay: set[str], basis: dict[int, str]) -> list[str]:
    errors: list[str] = []
    para: list[tuple[int, str]] = []
    fence = False

    def flush() -> None:
        if not para:
            return
        last = para[-1][0]
        # 依据行在段落最后一行之后：取这段之后第一个依据
        nxt = min((n for n in basis if n > last), default=None)
        self_made = nxt is not None and basis[nxt].strip() == SELF_MADE and nxt - last <= 2
        if not self_made:
            for no, line in para:
                probe = re.sub(r"\$\$.+?\$\$|(?<!\\)\$(?!\$)(?:\\.|[^$\n])+?(?<!\\)\$", " ", line)
                probe = re.sub(r"`[^`]*`|\[[^\]]*\]\([^)]*\)", " ", probe)
                for n in NUM_RE.findall(probe.replace(",", "")):
                    if n not in hay:
                        errors.append(f"第 {no} 行：数字 {n} 在 research/sources/ 里找不到；"
                                      "资料里没有的数字，要么删掉，要么改成自己构造的例子并写「依据：自行构造」")
        para.clear()

    for no, line in lines:
        s = line.strip()
        if s.startswith("```"):
            fence = not fence
            flush()
            continue
        if fence or s.startswith(("|", ":::", "#", "$$")):
            flush()
            continue
        if not s:
            flush()
        else:
            para.append((no, line))
    flush()
    return errors


# ---------- 正文 ----------

def check(md: Path, wiki_root: Path | None = None) -> list[str]:
    """返回全部问题，每条一行。生成器本身的格式错误由它自己报，这里不重复。"""
    wiki_root = wiki_root or REPO_ROOT / "wiki"
    errors: list[str] = []
    text = md.read_text(encoding="utf-8")
    try:
        doc = build.parse(text)
    except SystemExit as e:
        return [f"格式：{e}"]

    out_dir = md.parent.parent
    if not NAME_RE.fullmatch(out_dir.name):
        errors.append(f"目录名：{out_dir.name} 要用小写英文、数字和连字符（guides/concept.md 步骤 4）")

    meta = doc["meta"]
    for topic in [t.strip() for t in re.split(r"[,，]", meta["topics"]) if t.strip()]:
        if topic not in vocab.ALLOWED_TOPICS:
            errors.append(f"主题：「{topic}」不在词表里（可选：{'、'.join(vocab.ALLOWED_TOPICS)}）")
    if re.search(r"[,，]", meta["tag"]):
        errors.append(f"标签：只能写一个，现在是「{meta['tag']}」")
    elif meta["tag"] not in vocab.ALLOWED_TAGS:
        errors.append(f"标签：「{meta['tag']}」不在词表里（可选：{'、'.join(vocab.ALLOWED_TAGS)}）")

    for ref in doc["refs"]:
        if not (URL_RE.search(ref["text"]) or ISBN_RE.search(ref["text"])):
            errors.append(f"第 {ref['no']} 行：资料要带链接，没有网址的教材写 ISBN（guides/concept.md「资料」）")

    # 短页：恰好三节，不放折叠块、代码块、表格。标题文字不查，合不合适交给审大纲
    short_heads = [s for _no, s in ((n, l.strip()) for n, l in doc["short"]) if s.startswith("### ")]
    if len(short_heads) != SHORT_SECTIONS:
        errors.append(f"短页：要恰好 {SHORT_SECTIONS} 节（是什么、为什么需要、怎么做，标题与概念相关），"
                      f"现在是 {len(short_heads)} 节（guides/concept.md「短页」）")
    for no, line in doc["short"]:
        s = line.strip()
        if s.startswith(":::"):
            errors.append(f"第 {no} 行：短页不放折叠块（guides/concept.md「短页」）")
        elif s.startswith("```"):
            errors.append(f"第 {no} 行：短页不放代码块（guides/concept.md「短页」）")
        elif s.startswith("|"):
            errors.append(f"第 {no} 行：短页不放表格（guides/concept.md「短页」）")

    if doc["chapters"][-1][1] != LAST_CHAPTER:
        errors.append(f"第 {doc['chapters'][-1][0]} 行：长页最后一章的标题写「{LAST_CHAPTER}」，"
                      f"现在是「{doc['chapters'][-1][1]}」（guides/concept.md「长页」）")
    if len(doc["chapters"]) < 2:
        errors.append("长页：除「适用条件与边界」外，至少还要有一章讲机制")

    # 链接：正文不写网址；站内链接只指向现行的概念页
    for no, line in outside_fences(body_lines(doc)):
        probe = re.sub(r"\$\$.+?\$\$|(?<!\\)\$(?!\$)(?:\\.|[^$\n])+?(?<!\\)\$", " ", line)
        if URL_RE.search(probe):
            errors.append(f"第 {no} 行：正文里不写外部网址，出处放进「资料」（guides/concept.md「格式」）")
        for href in LINK_RE.findall(probe):
            m = re.fullmatch(r"wiki/([\w.-]+)(?:/(?:index|overview)\.html)?(?:#.*)?", href)
            if href.startswith(("assets/", "#")):
                continue
            if not m:
                errors.append(f"第 {no} 行：链接 {href} 只能写 wiki/<页面名>（guides/concept.md「格式」）")
                continue
            page = m.group(1)
            if not (wiki_root / page / "index.html").exists():
                errors.append(f"第 {no} 行：链接的页面 wiki/{page} 不存在，先按指南生成它")
                continue
            ptype, status = page_info(wiki_root, page)
            if ptype != "concept" or status != "current":
                errors.append(f"第 {no} 行：wiki/{page} 不是现行的概念页（类型 {ptype}，状态 {status}），"
                              "不能作前置概念链接")

    errors += check_basis(doc, md.parent)
    errors += check_numbers(doc, md.parent)

    # 依据必须单独成行：接在句子后面的「依据：」不会被生成器取出，会留在页面上
    for no, line in outside_fences(body_lines(doc)):
        if BASIS_MID_RE.search(line):
            errors.append(f"第 {no} 行：「依据：」要单独写成一行，放在这段正文的下一行，"
                          "不能接在句子后面，否则它会出现在页面上（guides/concept/content.md「怎么写」）")

    # 字数与代码块
    ctx = build.Ctx(out_dir, wiki_root)
    try:
        _body, _short, stats = build.render(doc, ctx)
    except SystemExit as e:
        return errors + [f"格式：{e}"]
    if stats["long"] > build.LONG_LIMIT:
        errors.append(f"字数：长页折叠块外 {stats['long']} 字，超出上限 {build.LONG_LIMIT}；"
                      "把推导、其余公式、例子过程、实验条件收进折叠块，或删掉（guides/concept.md「长页」）")
    if stats["short"] > build.SHORT_LIMIT:
        errors.append(f"字数：短页 {stats['short']} 字，超出上限 {build.SHORT_LIMIT}（guides/concept.md「短页」）")

    lines = text.splitlines()
    fences = ctx.fences
    for k, (no, lang, code) in enumerate(fences):
        if lang == "text":
            prev = fences[k - 1] if k else None
            if not prev or prev[1] != "python" or not _adjacent(lines, prev, no):
                errors.append(f"第 {no} 行：text 代码块只用来写 python 代码的输出，要紧跟在它后面"
                              "（guides/concept.md「格式」）")
        elif lang == "python":
            nxt = fences[k + 1] if k + 1 < len(fences) else None
            if not nxt or nxt[1] != "text" or not _adjacent(lines, (no, lang, code), nxt[0]):
                errors.append(f"第 {no} 行：python 代码后面要紧跟一个写运行输出的 text 代码块，"
                              "否则无法核对（guides/concept.md「格式」）")
        else:
            errors.append(f"第 {no} 行：代码块只能是 python 加输出块，现在是 {lang}；"
                          "换成文字或表格来写例子（guides/concept.md「格式」）")
    return errors


def _adjacent(lines: list[str], fence: tuple[int, str, str], next_no: int) -> bool:
    """fence 的结尾和下一个代码块的开头之间只有空行。"""
    no, _lang, code = fence
    end = no + code.count("\n") + 2 if code else no + 1
    return all(not l.strip() for l in lines[end:next_no - 1])


# ---------- 大纲 ----------

def check_outline(md: Path) -> list[str]:
    """大纲：长页和短页各一份，每个节点写五样：标题、写什么、类型、字数、依据。

    格式：
        ## 长页
        - 开头｜写什么｜类型｜字数｜依据
        - ### 章标题｜写什么｜类型｜字数｜依据
        - #### 小节标题｜写什么｜类型｜字数｜依据
        ## 短页
        - ### 节标题｜写什么｜类型｜字数｜依据
    """
    errors: list[str] = []
    text = md.read_text(encoding="utf-8")
    sections: dict[str, list[tuple[int, str]]] = {}
    cur = None
    for no, line in enumerate(text.splitlines(), 1):
        if line.startswith("## "):
            cur = line[3:].strip()
            sections[cur] = []
        elif cur and line.strip():
            sections[cur].append((no, line.strip()))
    for name in ("长页", "短页"):
        if name not in sections:
            errors.append(f"大纲：缺少「## {name}」一节")
    if errors:
        return errors

    sources = md.parent / "sources"
    cache: dict = {}
    totals = {"长页": 0, "短页": 0}
    for name in ("长页", "短页"):
        nodes = 0
        shorts = 0
        last_title = ""
        for no, line in sections[name]:
            if not line.startswith("- "):
                continue
            parts = [p.strip() for p in line[2:].split("｜")]
            if len(parts) != 5:
                errors.append(f"第 {no} 行：节点要写成「标题｜写什么｜类型｜字数｜依据」五样，现在是 {len(parts)} 样")
                continue
            title, what, kind, words, basis = parts
            nodes += 1
            if not what:
                errors.append(f"第 {no} 行：「写什么」不能空，一句话写这一节回答什么")
            elif len(what) > 80:
                errors.append(f"第 {no} 行：「写什么」只写一句话（80 字内），现在 {len(what)} 字；"
                              "大纲里不写成句的正文")
            if kind not in NODE_TYPES:
                errors.append(f"第 {no} 行：类型是 {'、'.join(NODE_TYPES)} 之一，现在是「{kind}」")
            wm = re.fullmatch(r"(\d+)", words)
            if not wm:
                errors.append(f"第 {no} 行：字数写成一个整数，现在是「{words}」")
            else:
                totals[name] += int(wm.group(1)) if kind != "折叠块" else 0
            if basis.strip() != SELF_MADE:
                for item in re.split(r"[;；]", basis):
                    if item.strip():
                        why = check_basis_item(item, sources, cache)
                        if why:
                            errors.append(f"第 {no} 行：依据对不上：{why}")
                if not basis.strip():
                    errors.append(f"第 {no} 行：依据不能空，写文件和位置，或写「{SELF_MADE}」")
            if name == "短页" and title.startswith("### "):
                shorts += 1
            last_title = title.lstrip("# ").strip()
        if nodes == 0:
            errors.append(f"大纲：「{name}」下没有节点")
        if name == "短页" and shorts != SHORT_SECTIONS:
            errors.append(f"大纲：短页要恰好 {SHORT_SECTIONS} 节（标题以「### 」开头），现在是 {shorts} 节")
        if name == "长页" and last_title != LAST_CHAPTER:
            errors.append(f"大纲：长页最后一个节点的标题写「{LAST_CHAPTER}」，现在是「{last_title}」")
    if totals["长页"] > build.LONG_LIMIT:
        errors.append(f"大纲：长页折叠块外的字数预算合计 {totals['长页']}，超出上限 {build.LONG_LIMIT}")
    if totals["短页"] > build.SHORT_LIMIT:
        errors.append(f"大纲：短页的字数预算合计 {totals['短页']}，超出上限 {build.SHORT_LIMIT}")
    return errors


# ---------- 自检 ----------

def _selftest() -> int:
    good = "\n".join([
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
    snap = ("# 第 1 章 玩具算子\n\n玩具算子把输入乘 2。输入取 3.5 时输出 7.0。\n"
            "它只对数值输入成立。\n\n## 1.2 另一节\n\n更多内容。\n")
    cases = [
        ("通过的页面没有报错", good, None),
        ("资料没有链接", good.replace("，[示例](https://example.com/book)", ""), "资料要带链接"),
        ("短页只有两节", good.replace("### 怎么用它\n\n读入一个数，输出它的两倍。\n依据：book.md L3-L4\n\n", ""),
         "要恰好 3 节"),
        ("短页有四节", good.replace("## 长页", "### 多出的一节\n\n内容。\n依据：自行构造\n\n## 长页", 1),
         "要恰好 3 节"),
        ("短页标题随便取也不报错", good.replace("它是什么", "完全不同的标题"), None),
        ("短页放折叠块", good.replace("读入一个数，输出它的两倍。",
                               "读入一个数。\n\n::: 详情\n内容\n:::"), "短页不放折叠块"),
        ("短页放表格", good.replace("读入一个数，输出它的两倍。",
                              "|a|b|\n|---|---|\n|1|2|"), "短页不放表格"),
        ("最后一章标题不对", good.replace("### 适用条件与边界", "### 什么时候不成立"), "最后一章"),
        ("正文写外部网址", good.replace("输出是输入的两倍", "输出是输入的两倍（https://a.com）"),
         "不写外部网址"),
        ("链接的页面不存在", good.replace("wiki/prereq", "wiki/nope"), "格式"),
        ("链接的不是概念页", good, "不是现行的概念页"),
        ("python 后面没有输出块", good.replace("```text\n2\n```\n", ""), "要紧跟"),
        ("其他语言的代码块", good.replace("```python\nprint(2 * 1)\n```\n\n```text\n2\n```", "```c\nint x;\n```"),
         "只能是 python"),
        ("长页超字数", good.replace("开头一段。", "字" * (build.LONG_LIMIT + 1)), "长页折叠块外"),
        ("短页超字数", good.replace("用来演示。", "字" * (build.SHORT_LIMIT + 1)), "短页"),
        ("依据的文件不存在", good.replace("依据：book.md 第 1 章", "依据：nope.md 第 1 章", 1), "不在 research/sources/ 下"),
        ("依据的章节找不到", good.replace("依据：book.md 第 1 章", "依据：book.md 第 9 章", 1), "里没有标题含"),
        ("依据的行号越界", good.replace("book.md L3-L4", "book.md L3-L99"), "只有"),
        ("依据写成标题的一段文字", good.replace("book.md 第 1 章", "book.md 玩具算子", 1), None),
        ("一段写两个依据", good.replace("依据：book.md 第 1 章", "依据：book.md 第 1 章；book.md 1.2 另一节", 1), None),
        ("整页没有依据", "\n".join(l for l in good.split("\n") if not l.startswith("依据：")), "全文没有一行"),
        ("依据接在句子后面", good.replace("玩具算子把输入乘 2。\n依据：book.md 第 1 章", "玩具算子把输入乘 2。依据：book.md 第 1 章"),
         "要单独写成一行"),
        ("长页里依据接在句子后面", good.replace("只对数值输入成立。\n依据：book.md 第 1 章", "只对数值输入成立。依据：book.md 第 1 章"),
         "要单独写成一行"),
        ("正文里自然出现的「依据：」不报", good.replace("只对数值输入成立。", "判断的依据：资料里写得很清楚。"), None),
        ("代码块里的依据不查", good.replace("print(2 * 1)", "print(2 * 1)  # 依据：无"), None),
        ("数字在资料里找不到", good.replace("取 3.5 时输出 7.0", "取 3.5 时输出 8.0"), "数字 8.0"),
        ("自行构造的段落数字不查", good.replace("用来演示。", "取 4242 试一下。"), None),
        ("数字在公式里不查", good.replace("开头一段。", "开头一段，$x=9999$。"), None),
        ("短页有一段没写依据", good.replace("用来演示。\n依据：自行构造\n", "用来演示。\n"), "后面没有「依据：」行"),
        ("长页有一段没写依据", good.replace("只对数值输入成立。\n依据：book.md 第 1 章", "只对数值输入成立。"),
         "后面没有「依据：」行"),
        ("引导句和后面的列表共用一行依据", good.replace("只对数值输入成立。",
                                             "分两步：\n\n1. 读入。\n2. 输出。"), None),
        ("小整数不能靠子串命中", good.replace("开头一段。", "开头一段，共有 7 个。"), "数字 7"),
    ]
    results: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory() as tmp:
        wiki = Path(tmp) / "wiki"
        (wiki / "prereq").mkdir(parents=True)
        (wiki / "prereq" / "index.html").write_text('<meta name="dojo:type" content="concept">', encoding="utf-8")
        (wiki / "oldie").mkdir()
        (wiki / "oldie" / "index.html").write_text(
            '<meta name="dojo:type" content="paper"><meta name="dojo:status" content="outdated">', encoding="utf-8")
        research = wiki / "toy" / "research"
        (research / "sources").mkdir(parents=True)
        (research / "sources" / "book.md").write_text(snap, encoding="utf-8")
        md = research / "content.md"
        for name, doc, want in cases:
            if name == "链接的不是概念页":
                doc = good.replace("wiki/prereq", "wiki/oldie")
            md.write_text(doc, encoding="utf-8")
            errs = check(md, wiki)
            if want is None:
                results.append((name, not errs))
                if errs:
                    print("    意外报错：", errs)
            else:
                results.append((name, any(want in e for e in errs)))

        bad = wiki / "Bad_Name" / "research"
        (bad / "sources").mkdir(parents=True)
        (bad / "sources" / "book.md").write_text(snap, encoding="utf-8")
        (bad / "content.md").write_text(good, encoding="utf-8")
        results.append(("目录名不合规应当报错", any("目录名" in e for e in check(bad / "content.md", wiki))))

        # 没有快照目录：依据与数字都无从核对
        nosrc = wiki / "nosrc" / "research"
        nosrc.mkdir(parents=True)
        (nosrc / "content.md").write_text(good, encoding="utf-8")
        results.append(("没有 sources 时报错", any("没有快照" in e for e in check(nosrc / "content.md", wiki))))

        # 大纲
        ol_good = "\n".join([
            "# 大纲", "", "## 长页", "",
            "- 开头｜讲清它是什么｜正文｜200｜book.md 第 1 章",
            "- ### 定义｜输出等于输入的两倍｜正文｜300｜book.md 第 1 章",
            "- #### 一个例子｜取 3.5 走一遍｜例子｜200｜自行构造",
            "- ### 适用条件与边界｜只对数值输入成立｜正文｜200｜book.md L3-L4", "",
            "## 短页", "",
            "- ### 它是什么｜一句话讲定义｜正文｜100｜book.md 第 1 章",
            "- ### 为什么要它｜讲动机｜正文｜100｜自行构造",
            "- ### 怎么用它｜讲用法｜正文｜100｜book.md 第 1 章",
        ])
        ocases = [
            ("通过的大纲没有报错", ol_good, None),
            ("大纲缺短页", ol_good.split("## 短页")[0], "缺少「## 短页」"),
            ("节点只有四样", ol_good.replace("｜正文｜200｜book.md 第 1 章", "｜正文｜book.md 第 1 章", 1), "五样"),
            ("类型不对", ol_good.replace("｜例子｜", "｜图｜"), "类型是"),
            ("字数不是整数", ol_good.replace("｜300｜", "｜三百｜"), "整数"),
            ("写什么超过一句话", ol_good.replace("输出等于输入的两倍", "字" * 90), "80 字"),
            ("依据的文件不存在", ol_good.replace("book.md 第 1 章", "x.md 第 1 章", 1), "不在 research/sources/ 下"),
            ("依据的章节找不到", ol_good.replace("book.md 第 1 章", "book.md 第 7 章", 1), "里没有标题含"),
            ("依据为空", ol_good.replace("｜自行构造", "｜", 1), "依据不能空"),
            ("短页不是三节", ol_good.replace("- ### 怎么用它｜讲用法｜正文｜100｜book.md 第 1 章", ""), "恰好 3 节"),
            ("长页最后一节不对", ol_good.replace("适用条件与边界", "什么时候不成立"), "最后一个节点"),
            ("长页预算超限", ol_good.replace("｜300｜", f"｜{build.LONG_LIMIT}｜"), "超出上限"),
            ("折叠块不计入预算", ol_good.replace("｜例子｜200｜", f"｜折叠块｜{build.LONG_LIMIT}｜"), None),
        ]
        omd = research / "outline.md"
        for name, doc, want in ocases:
            omd.write_text(doc, encoding="utf-8")
            errs = check_outline(omd)
            if want is None:
                results.append((name, not errs))
                if errs:
                    print("    意外报错：", errs)
            else:
                results.append((name, any(want in e for e in errs)))

    failed = [n for n, ok in results if not ok]
    for n, ok in results:
        print(("  OK  " if ok else "  FAIL") + " " + n)
    if failed:
        print(f"检查脚本自检失败：{len(failed)} 项")
        return 1
    print(f"检查脚本自检通过：{len(results)} 项")
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="概念页的机械检查：大纲格式与预算、正文的依据与结构")
    ap.add_argument("md", nargs="?", type=Path, help="wiki/<name>/research/content.md，加 --outline 时是 outline.md")
    ap.add_argument("--outline", action="store_true", help="检查大纲 outline.md")
    ap.add_argument("--selftest", action="store_true", help="用虚构数据自检本脚本")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(_selftest())
    if a.md is None:
        ap.error("需要 content.md 或 outline.md 的路径")
    if not a.md.exists():
        raise SystemExit(f"文件不存在：{a.md}")
    problems = check_outline(a.md) if a.outline else check(a.md)
    for p in problems:
        print(p)
    label = "大纲检查" if a.outline else "检查"
    print(f"{label}：{len(problems)} 个问题" if problems else f"{label}通过")
    raise SystemExit(1 if problems else 0)
