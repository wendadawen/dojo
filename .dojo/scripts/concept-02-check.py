#!/usr/bin/env python3
"""概念页的机械检查：读 research/concept.md，只查结构、字数、链接、代码块、资料与摘录。

用法（从仓库根运行）：
    python3 .dojo/scripts/concept-02-check.py wiki/<name>/research/concept.md
    python3 .dojo/scripts/concept-02-check.py wiki/<name>/research/concept.md --hash
    python3 .dojo/scripts/concept-02-check.py wiki/<name>/research/concept.md --gate
    python3 .dojo/scripts/concept-02-check.py --selftest

规则全部报错，每条写明位置和原因，一次报完。语义上的判断（事实、范围、重复、文风、读不读得通）
不在这里，交给 guides/concept/review.md 里的审查者。concept-01-build.py 生成完页面后会自动运行本脚本。
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
import tempfile
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


def _load(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


build = _load("concept-01-build.py", "concept_build")
vocab = _load("ci-01-validate.py", "ci_validate")

SHORT_HEADINGS = ["是什么", "为什么需要", "大概怎么工作"]
LAST_CHAPTER = "适用条件与边界"
NAME_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
URL_RE = re.compile(r"https?://", re.I)
LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)\s]+)\)")
ISBN_RE = re.compile(r"ISBN[\s:：-]*[0-9Xx-]{10,17}")
REVIEW_FILE_RE = re.compile(r"review-(\d+)\.md")
GATE_SECTIONS = ["摘录核验", "事实 1", "事实 2", "复算", "缺漏", "范围", "重复", "文风",
                 "读者 1", "读者 2", "处理"]


def _norm(text: str) -> str:
    """摘录与快照比对前的规范化：全角半角统一、去掉空白、列表符号、连字符断行和 LaTeX 记号。

    只消除抽取原文时带进来的版式噪声，不改词和数字，所以改了词或数字的摘录仍然对不上。
    """
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\\(?:mathrm|mathbf|mathcal|text|operatorname)\{([^}]*)\}", r"\1", text)
    text = text.replace("$", "")
    return re.sub(r"[\s\\{}_^\u2022]", "", text)


def check_excerpts(doc: dict, research: Path) -> list[str]:
    """每条摘录都要能在 research/sources/ 下的快照里逐字找到（规范化之后）。"""
    sources = research / "sources"
    files = sorted(p for p in sources.glob("*") if p.is_file()) if sources.is_dir() else []
    if not files:
        return ["摘录：research/sources/ 下没有原文快照。取原文时同时存一份，"
                "摘录才能对着它核对（guides/concept.md「资料与摘录」）"]
    haystack = "".join(_norm(p.read_text(encoding="utf-8", errors="ignore")) for p in files)
    errors = []
    for ref in doc["refs"]:
        for no, _pos, text in ref["excerpts"]:
            key = _norm(text)
            if key and key not in haystack:
                head = max((m for m in range(len(key), 8, -1) if key[:m] in haystack), default=0)
                where = (f"前 {head} 个字符吻合，之后对不上" if head
                         else "开头就对不上")
                errors.append(f"第 {no} 行：摘录在 research/sources/ 的快照里找不到（{where}）。"
                              "摘录要逐字抄，不能合并不相邻的句子，也不能改词和数字")
    return errors


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def body_lines(doc: dict) -> list[tuple[int, str]]:
    """短页和长页的全部行（含折叠块），不含资料。"""
    out = list(doc["short"]) + list(doc["opening"])
    for _no, _name, body in doc["chapters"]:
        out += body
    return out


def outside_fences(lines: list[tuple[int, str]]):
    """逐行产出代码块之外的行。"""
    fence = False
    for no, line in lines:
        if line.strip().startswith("```"):
            fence = not fence
            continue
        if not fence:
            yield no, line


def page_info(wiki_root: Path, page: str) -> tuple[str, str]:
    """读 wiki/<page>/index.html 的 dojo:type 和 dojo:status。"""
    text = (wiki_root / page / "index.html").read_text(encoding="utf-8")

    def meta(key: str) -> str:
        m = re.search(rf'<meta name="{key}" content="([^"]*)"', text)
        return m.group(1) if m else ""

    return meta("dojo:type"), meta("dojo:status") or "current"


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
    # 页面目录名
    if not NAME_RE.fullmatch(out_dir.name):
        errors.append(f"目录名：{out_dir.name} 要用小写英文、数字和连字符（页面名规则，见 guides/concept.md 步骤 3）")

    # 页面信息
    meta = doc["meta"]
    for topic in [t.strip() for t in re.split(r"[,，]", meta["topics"]) if t.strip()]:
        if topic not in vocab.ALLOWED_TOPICS:
            errors.append(f"主题：「{topic}」不在词表里（可选：{'、'.join(vocab.ALLOWED_TOPICS)}）")
    if re.search(r"[,，]", meta["tag"]):
        errors.append(f"标签：只能写一个，现在是「{meta['tag']}」")
    elif meta["tag"] not in vocab.ALLOWED_TAGS:
        errors.append(f"标签：「{meta['tag']}」不在词表里（可选：{'、'.join(vocab.ALLOWED_TAGS)}）")

    # 资料与摘录
    for ref in doc["refs"]:
        if not (re.search(r"https?://", ref["text"]) or ISBN_RE.search(ref["text"])):
            errors.append(f"第 {ref['no']} 行：资料要带链接，没有网址的教材写 ISBN（guides/concept.md「资料与摘录」）")
        if not ref["excerpts"]:
            errors.append(f"第 {ref['no']} 行：这条资料下面没有摘录，每条至少一条（guides/concept.md「资料与摘录」）")

    errors += check_excerpts(doc, md.parent)

    # 短页：三节、无折叠块/代码块/表格
    short_heads = [s[3:].strip() for _no, s in
                   ((n, l.strip()) for n, l in doc["short"]) if s.startswith("### ")]
    if short_heads != SHORT_HEADINGS:
        errors.append(f"短页：三节依次是「{'」「'.join(SHORT_HEADINGS)}」，现在是{short_heads or '没有小标题'}"
                      "（guides/concept.md「短页」）")
    for no, line in doc["short"]:
        s = line.strip()
        if s.startswith(":::"):
            errors.append(f"第 {no} 行：短页不放折叠块（guides/concept.md「短页」）")
        elif s.startswith("```"):
            errors.append(f"第 {no} 行：短页不放代码块（guides/concept.md「短页」）")
        elif s.startswith("|"):
            errors.append(f"第 {no} 行：短页不放表格（guides/concept.md「短页」）")

    # 长页最后一章
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

    # 生成并取字数、代码块
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

    # 代码块：只有 python 代码加紧跟的 text 输出块
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
    end = no + code.count("\n") + 2 if code else no + 1  # 开头行 + 代码行 + 结尾行
    between = lines[end:next_no - 1]
    return all(not l.strip() for l in between)


# ---------- 门禁 ----------

def gate(md: Path) -> list[str]:
    """最后一轮审查记录必须通过，且审查之后 concept.md 没有改过。"""
    research = md.parent
    rounds = sorted((int(m.group(1)), p) for p in research.glob("review-*.md")
                    if (m := REVIEW_FILE_RE.fullmatch(p.name)))
    if not rounds:
        return ["门禁：research/ 下没有 review-N.md，还没有审查记录"]
    n, path = rounds[-1]
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    m = re.search(r"^-\s*concept-sha256:\s*([0-9a-f]{64})\s*$", text, re.M)
    if not m:
        errors.append(f"门禁：{path.name} 缺少「- concept-sha256: <哈希>」")
    elif m.group(1) != sha256(md):
        errors.append(f"门禁：审查之后 concept.md 改过（{path.name} 记的是 {m.group(1)[:12]}…，"
                      f"现在是 {sha256(md)[:12]}…），改动没有被审过，重新审一轮")
    r = re.search(r"^-\s*结果:\s*(\S+)\s*$", text, re.M)
    if not r or r.group(1) != "通过":
        errors.append(f"门禁：最后一轮 {path.name} 的结果不是「通过」")
    heads = set(re.findall(r"^##\s+(.+?)\s*$", text, re.M))
    for name in GATE_SECTIONS:
        if name not in heads:
            errors.append(f"门禁：{path.name} 缺少「## {name}」一节")
    if re.search(r"未返回", text):
        errors.append(f"门禁：{path.name} 里有审查者「未返回」，补派后再记录")
    return errors


# ---------- 自检 ----------

def _selftest() -> int:
    good = "\n".join([
        "# 玩具算子", "",
        "- 描述：自检页。", "- 摘要：自检，与任何真实概念无关。",
        "- 主题：其他领域", "- 标签：通用概念", "",
        "## 资料", "",
        "- 虚构教材，第 1 版，[示例](https://example.com/book)",
        "  - 摘录（第 1 章）：The toy operator doubles its input.", "",
        "## 短页", "",
        "### 是什么", "", "玩具算子把输入乘 2。", "",
        "### 为什么需要", "", "用来演示。", "",
        "### 大概怎么工作", "", "读入一个数，输出它的两倍。", "",
        "## 长页", "", "开头一段。", "",
        "### 定义", "", "输出是输入的两倍，前置见[前置页](wiki/prereq)。", "",
        "```python", "print(2 * 1)", "```", "", "```text", "2", "```", "",
        "### 适用条件与边界", "", "只对数值输入成立。",
    ])
    cases = [
        ("通过的页面没有报错", good, None),
        ("摘录改了一个词", good.replace("doubles its input", "triples its input"), "快照里找不到"),
        ("摘录改了一个数字", good.replace("The toy operator doubles its input.",
                                   "The output is 2x, where x = 3.6 here."), "快照里找不到"),
        ("摘录拼接了不相邻的两句", good.replace("The toy operator doubles its input.",
                                       "The toy operator doubles its input. Second claim: the output is 2x"),
         "前"),
        ("摘录跨行断字仍能对上", good.replace("The toy operator doubles its input.",
                                      "The toy operator doubles its input. It is defined only for numbers."),
         None),
        ("没有快照", good, "没有原文快照"),
        ("资料没有链接", good.replace("，[示例](https://example.com/book)", ""), "资料要带链接"),
        ("资料没有摘录", good.replace("  - 摘录（第 1 章）：The toy operator doubles its input.\n", ""),
         "没有摘录"),
        ("摘录没有位置", good.replace("摘录（第 1 章）", "摘录（）"), "摘录"),
        ("主题不在词表", good.replace("其他领域", "占星术"), "主题"),
        ("标签不在词表", good.replace("通用概念", "不存在的标签"), "标签"),
        ("短页缺一节", good.replace("### 为什么需要\n\n用来演示。\n\n", ""), "三节依次是"),
        ("短页放折叠块", good.replace("读入一个数，输出它的两倍。",
                               "读入一个数。\n\n::: 详情\n内容\n:::"), "短页不放折叠块"),
        ("短页放表格", good.replace("读入一个数，输出它的两倍。",
                              "|a|b|\n|---|---|\n|1|2|"), "短页不放表格"),
        ("最后一章标题不对", good.replace("### 适用条件与边界", "### 什么时候不成立"), "最后一章"),
        ("只有一章", good.replace("### 定义", "### 适用条件与边界", 1)
         .replace("### 适用条件与边界\n\n只对", "### 其他\n\n只对"), "最后一章"),
        ("正文写外部网址", good.replace("输出是输入的两倍", "输出是输入的两倍（https://a.com）"),
         "不写外部网址"),
        ("链接的页面不存在", good.replace("wiki/prereq", "wiki/nope"), "格式"),
        ("链接的不是概念页", good, "不是现行的概念页"),
        ("python 后面没有输出块", good.replace("```text\n2\n```\n", ""), "要紧跟"),
        ("输出块不紧跟", good.replace("```\n\n```text", "```\n\n中间有一段。\n\n```text"), "要紧跟"),
        ("孤立的输出块", good.replace("```python\nprint(2 * 1)\n```\n\n", ""), "只用来写 python"),
        ("其他语言的代码块", good.replace("```python\nprint(2 * 1)\n```\n\n```text\n2\n```",
                                    "```c\nint x;\n```"), "只能是 python"),
        ("长页超字数", good.replace("开头一段。", "字" * (build.LONG_LIMIT + 1)), "长页折叠块外"),
        ("短页超字数", good.replace("用来演示。", "字" * (build.SHORT_LIMIT + 1)), "短页"),
    ]
    results: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        wiki = root / "wiki"
        (wiki / "prereq").mkdir(parents=True)
        (wiki / "prereq" / "index.html").write_text(
            '<meta name="dojo:type" content="concept">', encoding="utf-8")
        (wiki / "oldie").mkdir()
        (wiki / "oldie" / "index.html").write_text(
            '<meta name="dojo:type" content="paper"><meta name="dojo:status" content="outdated">',
            encoding="utf-8")
        research = wiki / "toy" / "research"
        (research / "sources").mkdir(parents=True)
        (research / "sources" / "book.txt").write_text(
            "Chapter 1.\nThe toy operator doubles its input. It is\nde-\nfined only for numbers.\n"
            "• Second claim: the output is 2x, where x = 3.5 here.\n", encoding="utf-8")
        md = research / "concept.md"
        for name, doc, want in cases:
            if name == "链接的不是概念页":
                doc = good.replace("wiki/prereq", "wiki/oldie")
            md.write_text(doc, encoding="utf-8")
            snap = research / "sources" / "book.txt"
            if name == "没有快照":
                saved = snap.read_text(encoding="utf-8"); snap.unlink()
                (research / "sources").rmdir()
            errs = check(md, wiki)
            if name == "没有快照":
                (research / "sources").mkdir(); snap.write_text(saved, encoding="utf-8")
            if name == "摘录跨行断字仍能对上":
                results.append((name, not any("快照" in e for e in errs)))
                continue
            if want is None:
                results.append((name, not errs))
                if errs:
                    print("    意外报错：", errs)
            else:
                results.append((name, any(want in e for e in errs)))

        # 目录名
        bad = wiki / "Bad_Name" / "research"
        bad.mkdir(parents=True)
        (bad / "concept.md").write_text(good, encoding="utf-8")
        results.append(("目录名不合规应当报错",
                        any("目录名" in e for e in check(bad / "concept.md", wiki))))

        # 门禁
        md.write_text(good, encoding="utf-8")
        results.append(("没有审查记录时门禁报错", any("没有审查记录" in e for e in gate(md))))
        sections = "\n".join(f"## {s}\n\n- 无\n" for s in GATE_SECTIONS)

        def review(name: str, digest: str, result: str, extra: str = "") -> None:
            (research / name).write_text(
                f"# 第 1 轮\n\n- concept-sha256: {digest}\n- 结果: {result}\n\n{sections}{extra}",
                encoding="utf-8")

        review("review-1.md", sha256(md), "通过")
        results.append(("记录齐全且哈希一致时门禁通过", not gate(md)))
        review("review-1.md", sha256(md), "未通过")
        results.append(("结果未通过时门禁报错", any("不是「通过」" in e for e in gate(md))))
        review("review-1.md", "0" * 64, "通过")
        results.append(("concept.md 改过时门禁报错", any("改过" in e for e in gate(md))))
        review("review-1.md", sha256(md), "通过", "\n事实 2：未返回\n")
        results.append(("有未返回时门禁报错", any("未返回" in e for e in gate(md))))
        (research / "review-1.md").write_text(
            f"# 第 1 轮\n\n- concept-sha256: {sha256(md)}\n- 结果: 通过\n\n## 摘录核验\n\n无\n",
            encoding="utf-8")
        results.append(("缺节时门禁报错", any("缺少「## 事实 1」" in e for e in gate(md))))
        review("review-2.md", sha256(md), "通过")
        (research / "review-1.md").write_text("坏的", encoding="utf-8")
        results.append(("只认最后一轮", not gate(md)))

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

    ap = argparse.ArgumentParser(description="概念页的机械检查与审查门禁")
    ap.add_argument("md", nargs="?", type=Path, help="wiki/<name>/research/concept.md")
    ap.add_argument("--hash", action="store_true", help="打印 concept.md 的 SHA-256，写进审查记录")
    ap.add_argument("--gate", action="store_true", help="检查最后一轮审查记录通过且 concept.md 未改")
    ap.add_argument("--selftest", action="store_true", help="用虚构数据自检本脚本")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(_selftest())
    if a.md is None:
        ap.error("需要 concept.md 的路径")
    if not a.md.exists():
        raise SystemExit(f"文件不存在：{a.md}")
    if a.hash:
        print(sha256(a.md))
        raise SystemExit(0)
    problems = gate(a.md) if a.gate else check(a.md)
    for p in problems:
        print(p)
    label = "门禁" if a.gate else "检查"
    print(f"{label}：{len(problems)} 个问题" if problems else f"{label}通过")
    raise SystemExit(1 if problems else 0)
