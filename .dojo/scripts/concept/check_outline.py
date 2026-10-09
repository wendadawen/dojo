#!/usr/bin/env python3
"""检查 outline.md：格式、字数预算、依据能不能对上资料。"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (  # noqa: E402
    LAST_CHAPTER, LONG_LIMIT, OUTLINE_ROUNDS, SELF_MADE, SHORT_LIMIT, SHORT_SECTIONS, SNAPSHOT,
    check_basis_list, final_check_recorded, report,
)


# ---------- 解析 ----------

def parse(md: Path) -> dict:
    """返回 {sections, nodes, bad, sources}。
    sections 是「## 长页」「## 短页」下的行；nodes 是按页分的节点列表，每个节点是
    (行号, 标题, 写什么, 字数, 依据)；格式不对的节点记在 bad 里。
    """
    sections: dict[str, list[tuple[int, str]]] = {}
    cur = None
    for no, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("## "):
            cur = line[3:].strip()
            sections[cur] = []
        elif cur and line.strip():
            sections[cur].append((no, line.strip()))
    nodes: dict[str, list[tuple]] = {"长页": [], "短页": []}
    bad: list[tuple[int, int]] = []
    for name in nodes:
        for no, line in sections.get(name, []):
            if not line.startswith("- "):
                continue
            parts = [p.strip() for p in line[2:].split("｜")]
            if len(parts) != 4:
                bad.append((no, len(parts)))
            else:
                nodes[name].append((no, *parts))
    return {"sections": sections, "nodes": nodes, "bad": bad, "sources": md.parent / "sources"}


# ---------- 规则：每条一个函数，返回问题列表 ----------

def has_both_pages(o: dict) -> list[str]:
    return [f"大纲：缺少「## {n}」一节" for n in ("长页", "短页") if n not in o["sections"]]


def node_has_four_parts(o: dict) -> list[str]:
    return [f"第 {no} 行：节点要写成「标题｜写什么｜字数｜依据」四样，现在是 {n} 样" for no, n in o["bad"]]


def each_page_has_nodes(o: dict) -> list[str]:
    return [f"大纲：「{n}」下没有节点" for n in ("长页", "短页") if not o["nodes"][n]]


def what_is_one_short_sentence(o: dict) -> list[str]:
    errors = []
    for page in o["nodes"].values():
        for no, _title, what, *_ in page:
            if not what:
                errors.append(f"第 {no} 行：「写什么」不能空，一句话写这一节回答什么")
            elif len(what) > 80:
                errors.append(f"第 {no} 行：「写什么」只写一句话（80 字内），现在 {len(what)} 字；大纲里不写成句的正文")
    return errors


def words_is_integer(o: dict) -> list[str]:
    return [f"第 {no} 行：字数写成一个整数，现在是「{words}」"
            for page in o["nodes"].values() for no, _t, _w, words, _b in page if not re.fullmatch(r"\d+", words)]


def basis_matches_sources(o: dict) -> list[str]:
    errors = []
    for page in o["nodes"].values():
        for no, *_, basis in page:
            if not basis.strip():
                errors.append(f"第 {no} 行：依据不能空，写文件和位置，或写「{SELF_MADE}」")
            for why in check_basis_list(basis, o["sources"]):
                errors.append(f"第 {no} 行：依据对不上：{why}")
    return errors


def short_has_three_sections(o: dict) -> list[str]:
    n = sum(1 for _no, title, *_ in o["nodes"]["短页"] if title.startswith("### "))
    if o["nodes"]["短页"] and n != SHORT_SECTIONS:
        return [f"大纲：短页要恰好 {SHORT_SECTIONS} 节（标题以「### 」开头），现在是 {n} 节"]
    return []


def long_ends_with_boundary(o: dict) -> list[str]:
    if not o["nodes"]["长页"]:
        return []
    last = o["nodes"]["长页"][-1][1].lstrip("# ").strip()
    if last != LAST_CHAPTER:
        return [f"大纲：长页最后一个节点的标题写「{LAST_CHAPTER}」，现在是「{last}」"]
    return []


def budget_within_limit(o: dict) -> list[str]:
    """字数预算只算折叠块外的正文，折叠块不单独写预算。"""
    errors = []
    for name, limit in (("长页", LONG_LIMIT), ("短页", SHORT_LIMIT)):
        total = sum(int(words) for _no, _t, _w, words, _b in o["nodes"][name] if words.isdigit())
        if total > limit:
            errors.append(f"大纲：{name}的字数预算合计 {total}，超出上限 {limit}")
    return errors


def final_check_is_recorded(o: dict) -> list[str]:
    return final_check_recorded(o["sources"].parent, "outline", OUTLINE_ROUNDS)


RULES = [
    node_has_four_parts, each_page_has_nodes, what_is_one_short_sentence,
    words_is_integer, basis_matches_sources, short_has_three_sections,
    long_ends_with_boundary, budget_within_limit, final_check_is_recorded,
]


def check(md: Path) -> list[str]:
    o = parse(md)
    errors = has_both_pages(o)
    if errors:
        return errors  # 缺一页时，后面的规则没有意义
    for rule in RULES:
        errors += rule(o)
    return errors


# ---------- 自检 ----------

GOOD = "\n".join([
    "# 大纲", "", "## 长页", "",
    "- 开头｜讲清它是什么｜200｜book.md 第 1 章",
    "- ### 定义｜输出等于输入的两倍｜300｜book.md 第 1 章",
    "- #### 一个例子｜取 3.5 走一遍，过程折叠｜200｜自行构造",
    "- ### 适用条件与边界｜只对数值输入成立｜200｜book.md L3-L4", "",
    "## 短页", "",
    "- ### 它是什么｜一句话讲定义｜100｜book.md 第 1 章",
    "- ### 为什么要它｜讲动机｜100｜自行构造",
    "- ### 怎么用它｜讲用法｜100｜book.md 第 1 章",
])

# 每条规则一个坏输入，应当报出 want；want 为 None 的是不该报错的写法
CASES = [
    ("通过的大纲没有报错", GOOD, None),
    ("缺短页", GOOD.split("## 短页")[0], "缺少「## 短页」"),
    ("节点只有三样", GOOD.replace("｜200｜book.md 第 1 章", "｜book.md 第 1 章", 1), "四样"),
    ("还带着旧的类型一栏", GOOD.replace("｜200｜book.md 第 1 章", "｜正文｜200｜book.md 第 1 章", 1), "四样"),
    ("字数不是整数", GOOD.replace("｜300｜", "｜三百｜"), "整数"),
    ("写什么超过一句话", GOOD.replace("输出等于输入的两倍", "字" * 90), "80 字"),
    ("依据的文件不存在", GOOD.replace("book.md 第 1 章", "x.md 第 1 章", 1), "不在 research/sources/ 下"),
    ("依据的章节找不到", GOOD.replace("book.md 第 1 章", "book.md 第 7 章", 1), "里没有标题含"),
    ("依据为空", GOOD.replace("｜自行构造", "｜", 1), "依据不能空"),
    ("短页不是三节", GOOD.replace("- ### 怎么用它｜讲用法｜100｜book.md 第 1 章", ""), "恰好 3 节"),
    ("长页最后一节不对", GOOD.replace("适用条件与边界", "什么时候不成立"), "最后一个节点"),
    ("长页预算超限", GOOD.replace("｜300｜", f"｜{LONG_LIMIT}｜"), "超出上限"),
]


def selftest() -> int:
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        research = Path(tmp) / "wiki" / "toy" / "research"
        (research / "sources").mkdir(parents=True)
        (research / "sources" / "book.md").write_text(SNAPSHOT, encoding="utf-8")
        md = research / "outline.md"
        for name, text, want in CASES:
            md.write_text(text, encoding="utf-8")
            errs = check(md)
            ok = (not errs) if want is None else any(want in e for e in errs)
            if want is None and errs:
                print("    意外报错：", errs)
            results.append((name, ok))
        record = research / f"outline-review-{OUTLINE_ROUNDS}.md"
        md.write_text(GOOD, encoding="utf-8")
        record.write_text("## 处理\n全部采纳。\n", encoding="utf-8")
        results.append(("审到轮数上限没写最后一批改动核对", any("最后一批改动核对" in e for e in check(md))))
        record.write_text("## 最后一批改动核对\n无改动。\n", encoding="utf-8")
        results.append(("写了最后一批改动核对不报", not check(md)))
    return report(results, "大纲检查")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    if len(sys.argv) != 2 or not Path(sys.argv[1]).exists():
        raise SystemExit("用法：check_outline.py wiki/<name>/research/outline.md")
    problems = check(Path(sys.argv[1]))
    for p in problems:
        print(p)
    print(f"大纲检查：{len(problems)} 个问题" if problems else "大纲检查通过")
    raise SystemExit(1 if problems else 0)
