#!/usr/bin/env python3
"""数据流页的生成器：读计划文件 research/dataflow.md，生成页面 index.html。

用法（从仓库根运行）：
    python3 .dojo/scripts/dataflow/build_page.py wiki/<name>/research/dataflow.md
    python3 .dojo/scripts/dataflow/build_page.py --selftest

计划文件的格式见 guides/dataflow/plan.md。页面上的文字全部来自计划文件，生成器只做装配：
读模板、替换占位符、把视图数据写进页面。

页面骨架是 .dojo/templates/dataflow/index.html，唯一真相源。

数据结构（与引擎 libs/dataflow/dojo-flow.js 的契约，详见模板注释）
    view  = { id, label, title, nodes[], edges[], groups[]?, notes[]?, info[]? }
    info  = [{ title, rows: [[键, 值], ...] }]  说明面板，随视图切换
    node  = { id, name, shape?, detail?, kind, drill?, fold? }
    fold  = [{ title, text }]   算子的折叠，点标记打开；text 里可以有 $...$
    edge  = { from, to, label? }
    group = { label, members[], stroke? }
    note  = { at, text, dx?, dy? }
    kind：tensor | op | cache | port
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "site"))
from common import (  # noqa: E402
    KIND, META_KEYS, SHAPE_PLACEHOLDERS, SKIP_HEADS, SUB_SECTIONS, report,
)
from vocab import ALLOWED_TAGS, ALLOWED_TOPICS  # noqa: E402

TEMPLATE = HERE.parents[1] / "templates" / "dataflow" / "index.html"
REPO_ROOT = HERE.parents[2]

DRILL_RE = re.compile(r"^-\s*(\S+)\s+点开\s*(?:→|->)\s*(.+?)\s*$")
EDGE_RE = re.compile(r"^-\s*([^\s→]+?)\s*(?:→|->)\s*([^\s：:]+)\s*(?:[：:]\s*(.+?))?\s*$")


def fail(no: int, msg: str) -> None:
    raise SystemExit(f"第 {no} 行：{msg}" if no else msg)


def node(nid: str, name: str, shape: str | None = None, kind: str = "op",
         src: str | None = None, detail: str | None = None,
         drill: str | None = None) -> dict:
    """构造一个节点。src 只是构建期的标注（源码行号或权重键），供人复核，不写进页面。"""
    out = {"id": nid, "name": name, "kind": kind}
    if shape:
        out["shape"] = shape
    if detail:
        out["detail"] = detail
    if drill:
        out["drill"] = drill
    return out


def edge(a: str, b: str, label: str | None = None) -> dict:
    out = {"from": a, "to": b}
    if label:
        out["label"] = label
    return out


def view(vid: str, label: str, title: str, nodes: list[dict], edges: list[dict],
         groups: list[dict] | None = None, notes: list[dict] | None = None,
         info: list | None = None) -> dict:
    """info 与 config_groups 同形：(分组名, [(键, 值)])，渲染成随视图切换的说明面板。"""
    out = {"id": vid, "label": label, "title": title, "nodes": nodes, "edges": edges,
           "groups": groups or [], "notes": notes or []}
    if info:
        out["info"] = [{"title": t, "rows": [[k, v] for k, v in rows]} for t, rows in info]
    return out


# ---------- 从计划文件读 ----------

def split_cells(line: str) -> list[str]:
    """一行 Markdown 表格拆成单元格：$...$ 里的竖线不拆列，\\| 是字面竖线，
    <br> 换行，反引号去掉。"""
    body = line.strip().strip("|")
    masked: list[str] = []

    def hide(match: re.Match) -> str:
        masked.append(match.group(0))
        return f"\x00{len(masked) - 1}\x00"

    protected = re.sub(r"\$(?:\\.|[^$])*\$", hide, body)
    cells = re.split(r"(?<!\\)\|", protected)

    def restore(text: str) -> str:
        return re.sub(r"\x00(\d+)\x00", lambda m: masked[int(m.group(1))], text)

    return [re.sub(r"<br\s*/?>", "\n", restore(c).replace("\\|", "|")).replace("`", "").strip()
            for c in cells]


def read_table(rows: list[tuple[int, str]], need: list[str], where: str) -> list[tuple[int, dict]]:
    """表头按列名取值；缺列时报出表头所在行。"""
    if not rows:
        return []
    head_no, head = rows[0]
    names = split_cells(head)
    missing = [n for n in need if n not in names]
    if missing:
        fail(head_no, f"{where}的表头缺少列 {'、'.join(missing)}")
    out = []
    for no, line in rows[1:]:
        cells = split_cells(line)
        if all(re.fullmatch(r":?-{3,}:?", c) for c in cells if c):
            continue
        out.append((no, dict(zip(names, cells + [""] * (len(names) - len(cells))))))
    return out


def group_rows(rows: list[tuple[int, dict]]) -> list[tuple[str, list[tuple[str, str]]]]:
    groups: dict[str, list[tuple[str, str]]] = {}
    for _, r in rows:
        groups.setdefault(r["组"], []).append((r["名字"], r["值"]))
    return list(groups.items())


def parse(plan: Path) -> dict:
    """按 guides/dataflow/plan.md 的格式读计划文件，返回 page() 需要的参数。"""
    text = plan.read_text(encoding="utf-8")
    meta: dict[str, str] = {}
    sources: list[str] = []
    scope: list[str] = []
    config_rows: list[tuple[int, str]] = []
    views_raw: list[dict] = []
    section = sub = None
    in_fence = False
    for no, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if s.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not s:
            continue
        if s.startswith("# "):
            meta["title"] = s[2:].strip()
        elif s.startswith("## "):
            head = s[3:].strip()
            if head.lower() == "config.json":
                section, sub = "config", None
            elif head == "资料":
                section, sub = "sources", None
            elif head == "前提":
                section, sub = "scope", None
            else:
                section, sub = "view", None
                views_raw.append({"head": head, "no": no, "nodes": [], "edges": [],
                                  "drills": [], "panel": [], "folds": []})
        elif s.startswith("### "):
            sub = s[4:].strip()
            if section != "view" or sub not in SUB_SECTIONS:
                fail(no, f"不认识的小节「{sub}」，只能是 节点 / 连线 / 折叠 / 面板")
        elif section is None and s.startswith("-"):
            key, sep, value = s[1:].strip().partition("：")
            if not sep:
                key, sep, value = s[1:].strip().partition(":")
            if key.strip() in META_KEYS:
                meta[META_KEYS[key.strip()]] = value.strip()
        elif section == "sources" and s.startswith("-"):
            sources.append(s[1:].strip().replace("`", ""))
        elif section == "scope" and s.startswith("-"):
            scope.append(s[1:].strip().replace("`", ""))
        elif section == "config" and s.startswith("|"):
            config_rows.append((no, s))
        elif section == "view" and sub in ("节点", "面板", "折叠") and s.startswith("|"):
            views_raw[-1][{"节点": "nodes", "面板": "panel", "折叠": "folds"}[sub]].append((no, s))
        elif section == "view" and sub == "连线" and s.startswith("-"):
            m = DRILL_RE.match(s)
            if m:
                views_raw[-1]["drills"].append((no, m.group(1), m.group(2)))
                continue
            m = EDGE_RE.match(s)
            if not m:
                fail(no, "连线要写成「- a → b」或「- a → b：线上的字」")
            views_raw[-1]["edges"].append((no, m.group(1), m.group(2), m.group(3)))

    for zh, key in META_KEYS.items():
        if key not in meta:
            name = "模型名（# 标题）" if key == "title" else zh
            raise SystemExit(f"计划文件缺少：{name}")
    if not views_raw:
        raise SystemExit("计划文件里没有图（## <类名>.forward（...））")
    if not sources:
        raise SystemExit("计划文件缺少「资料」一节")
    if not scope:
        raise SystemExit("计划文件缺少「前提」一节")

    bad_topics = [t.strip() for t in re.split(r"[,，]", meta["topics"]) if t.strip() not in ALLOWED_TOPICS]
    if bad_topics:
        raise SystemExit("主题不在词表里：" + "、".join(bad_topics))
    if meta["tag"] not in ALLOWED_TAGS:
        raise SystemExit(f"标签「{meta['tag']}」不在词表里")

    used: set[str] = set()
    for v in views_raw:
        label = re.split(r"[（(]", v["head"], maxsplit=1)[0].strip()
        label = label[:-len(".forward")] if label.endswith(".forward") else label
        vid = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or "view"
        base, k = vid, 2
        while vid in used:
            vid, k = f"{base}-{k}", k + 1
        used.add(vid)
        v["label"], v["id"] = label, vid

    def find_view(name: str) -> str | None:
        for v in views_raw:
            short = re.split(r"[（(]", v["head"], maxsplit=1)[0].strip()
            if name in (v["head"], short, v["label"]):
                return v["id"]
        return None

    views = []
    for v in views_raw:
        where = f"图「{v['head']}」"
        nodes, ids = [], set()
        for no, r in read_table(v["nodes"], ["id", "类型", "第一行", "第二行", "第三行"],
                                where + "节点表"):
            nid = r["id"]
            if not nid or nid in ids:
                fail(no, f"id「{nid}」为空或在{where}里重复")
            if not re.fullmatch(r"[A-Za-z0-9_.\-]+", nid):
                fail(no, f"id「{nid}」只能用英文、数字、下划线、点、连字符")
            if r["类型"] not in KIND:
                fail(no, f"类型「{r['类型']}」只能是 算子 / 张量 / 缓存")
            ids.add(nid)
            nodes.append(node(nid, r["第一行"], r["第二行"] or None, KIND[r["类型"]],
                              src=r.get("源码") or None, detail=r["第三行"] or None))
        if not nodes:
            fail(v["no"], f"{where}没有节点表")
        by_id = {n["id"]: n for n in nodes}
        edges = []
        for no, a, b, label in v["edges"]:
            for end in (a, b):
                if end not in by_id:
                    fail(no, f"{where}里没有节点「{end}」")
            edges.append(edge(a, b, label))
        for no, nid, target in v["drills"]:
            if nid not in by_id:
                fail(no, f"{where}里没有节点「{nid}」")
            if by_id[nid].get("drill"):
                # 一个节点只留一条点开，第二条会静默覆盖前一条，报出来
                fail(no, f"{where}节点「{nid}」有不止一条点开，只留一条")
            tid = find_view(target)
            if tid is None:
                fail(no, f"点开的目标「{target}」不是任何一张图的标题")
            by_id[nid]["drill"] = tid
        for no, r in read_table(v["folds"], ["id", "名字", "内容"], where + "折叠表"):
            nid = r["id"]
            if nid not in by_id:
                fail(no, f"{where}里没有节点「{nid}」")
            if by_id[nid]["kind"] != "op":
                fail(no, f"折叠只写在算子上，「{nid}」不是算子")
            if r["名字"] not in ("公式", "符号", "说明"):
                fail(no, "折叠的名字只能是「公式」「符号」或「说明」")
            if not r["内容"]:
                fail(no, "折叠要写内容")
            by_id[nid].setdefault("fold", []).append({"title": r["名字"], "text": r["内容"]})
        info = group_rows(read_table(v["panel"], ["组", "名字", "值"], where + "面板"))
        title = re.split(r"[（(]", v["head"], maxsplit=1)[0].strip()
        views.append(view(v["id"], v["label"], title, nodes, edges, info=info))

    config = group_rows(read_table(config_rows, ["组", "名字", "值"], "config.json 表"))
    check_plan(views, config, scope, meta)
    return {"plan": plan, "meta": meta, "views": views, "config_groups": config or None,
            "config_note": scope}


def check_plan(views: list[dict], config_groups=None, config_note=None, meta=None) -> None:
    """跨图与形状的机械检查：同 id 必须同形状，算子和张量的形状要写全，公式外的下划线不转义。

    不同图里第一行可以不一样——一处写变量名、一处写产生它的表达式，都指同一个量；
    但形状必须一致。同 id 形状对不上，就是两个不同的量共用了 id。
    """
    errors: list[str] = []
    shapes: dict[str, str] = {}        # id -> 第一次出现的形状
    for v in views:
        for n in v["nodes"]:
            where = f"图「{v['label']}」/{n['id']}"
            shape = (n.get("shape") or "").strip()
            if n["kind"] == "tensor":
                if not shape or shape in SHAPE_PLACEHOLDERS:
                    errors.append(f"{where}：张量第二行要写数据类型和形状")
                elif n["id"] in shapes and shapes[n["id"]] != shape:
                    errors.append(f"{where}：id「{n['id']}」在别处是「{shapes[n['id']]}」，"
                                  f"这里是「{shape}」；不是同一个量就换一个 id")
                else:
                    shapes.setdefault(n["id"], shape)
            elif n["kind"] == "cache":
                if not shape or shape in SHAPE_PLACEHOLDERS:
                    errors.append(f"{where}：缓存第二行要写存的是什么")
            else:
                if "→" not in shape and "->" not in shape:
                    errors.append(f"{where}：算子第二行要写「原来的形状 → 变完的形状」")
                else:
                    for side in re.split(r"→|->", shape):
                        if not side.strip() or side.strip() in SHAPE_PLACEHOLDERS:
                            errors.append(f"{where}：算子第二行两边都要写完整形状，现在是「{shape}」")
                            break
            errors += _stray_underscore_escape(where, n)
            errors += _formula_in_explanation(where, n)
            if n["kind"] == "op":
                errors += _one_step_per_node(where, n)
                if (n.get("detail") or "").strip():
                    errors.append(f"{where}：算子第三行留空，较长说明写进折叠的「说明」")
        for row in v.get("info") or []:
            # 面板的每一格也查越界下划线（docstring 早就这么写，实际漏了）
            for name, value in row["rows"]:
                errors += _stray_underscore_escape(f"图「{v['label']}」面板/{row['title']}/{name}",
                                                   {"detail": f"{name}\n{value}"})
            if row["title"] == "公式":
                for name, value in row["rows"]:
                    if len(re.findall(r"\$[^$]*\$", value)) > 1:
                        errors.append(f"图「{v['label']}」面板/公式/{name}："
                                      "一条只放一个式子，几个式子分几条")
    # config.json 表与「前提」不在任一视图里，单独查越界下划线
    for title, rows in config_groups or []:
        for name, value in rows:
            errors += _stray_underscore_escape(f"config.json 面板/{title}/{name}",
                                               {"detail": f"{name}\n{value}"})
    for i, line in enumerate(config_note or [], 1):
        errors += _stray_underscore_escape(f"前提/第 {i} 条", {"detail": line})
    # 页面信息（标题、描述、摘要）都进页面 head，同样查越界下划线
    if meta:
        for key, zh in (("title", "标题"), ("description", "描述"), ("summary", "摘要")):
            if meta.get(key):
                errors += _stray_underscore_escape(f"页面信息/{zh}", {"detail": meta[key]})
    errors += _no_generic_class_figure(views)
    if errors:
        raise SystemExit("\n".join(errors))


def _stray_underscore_escape(where: str, node: dict) -> list[str]:
    """`\\_` 只在 $...$ 里有用（LaTeX 的下标分隔符）。公式外的 `\\_` 会原样显示成反斜杠，
    要写成普通下划线。折叠和面板的每一行都要查。"""
    texts = [node.get("shape"), node.get("detail"),
             *(s.get("text") for s in node.get("fold") or [])]
    errors = []
    for text in texts:
        if not text:
            continue
        for line in re.split(r"<br\s*/?>|\n", text):
            outside = re.sub(r"\$[^$\n]*\$", "", line)
            if "\\_" in outside:
                errors.append(f"{where}：下划线写成 \\_ 了，公式外用普通 _（{outside.strip()[:40]}）")
                break
    return errors


# 说明里允许引的「一个符号」：可带包裹（\mathbf、\mathcal）和下标/上标，没有运算符和函数
SYMBOL_RE = re.compile(
    r"^(?:\\[A-Za-z]+)?\{*[A-Za-z\\][^{}]*\}*"
    r"(?:_\{[^{}]*\}|\^\{[^{}]*\}|_[A-Za-z0-9]|\^[A-Za-z0-9])*$")


def _is_bare_symbol(tex: str) -> bool:
    """`$...$` 里只有一个符号（可带下标/上标和 \\mathbf 这类包裹）就返回 True。
    有空格、运算符、逗号、括号的，算公式。"""
    tex = tex.strip()
    if not tex or any(ch in tex for ch in " \t=+*/<>(),|[]"):
        return False
    if "\\" in tex and not re.match(r"^(\\[A-Za-z]+)" + r"\{?[A-Za-z\\]", tex):
        return False
    return bool(SYMBOL_RE.match(tex))


def _formula_in_explanation(where: str, node: dict) -> list[str]:
    """折叠的「说明」写权重形状、配置和一句说明，不重复公式——公式归「公式」那一行，
    算子和张量的节点本身已经画出这些量了。

    说明里引一个符号（`$I$`、`$\\mathbf{C}$`）不算公式，放行；写成带运算符的式子才算。"""
    errors = []
    for sec in node.get("fold") or []:
        if sec.get("title") != "说明":
            continue
        for tex in re.findall(r"\$([^$]+)\$", sec.get("text") or ""):
            if not _is_bare_symbol(tex):
                errors.append(f"{where}：折叠「说明」里不写公式，公式放进「公式」那一行")
                break
    return errors


# ---------- 写页面 ----------

# 只是把通用算子套在权重上的基础层，不单独成图
GENERIC_CLASSES = {
    "Linear", "ColumnParallelLinear", "RowParallelLinear", "RMSNorm",
    "Embedding", "ParallelEmbedding", "PatchEmbed", "MLP", "ParallelHead",
    # transformers 把通用层也按模型前缀命名，仍然只是套通用算子，一并列出
    "Qwen2MLP", "Qwen2RMSNorm", "Qwen3MLP", "Qwen3RMSNorm",
    "LlamaMLP", "LlamaRMSNorm", "MistralMLP", "MistralRMSNorm",
}


def _one_step_per_node(where: str, node: dict) -> list[str]:
    """一个节点一步：第一行里套了两层以上类调用（如 self.a(self.b(x))），要拆开。"""
    calls = re.findall(r"self\.([a-z_][a-z0-9_]*)\s*\(", node.get("name") or "")
    if len(calls) >= 2:
        return [f"{where}：一个节点一步，第一行套了 {'、'.join(calls)} 两个调用，拆开"]
    return []


def _no_generic_class_figure(views: list[dict]) -> list[str]:
    """基础层（Linear、RMSNorm 这类）不单独成图，写在使用它们的节点上。

    图标题是「<类名>.forward」（如 `vision.Block.forward`），取 `.forward` 前的那一段当类名；
    再去掉模块前缀与模型名前缀后比 GENERIC_CLASSES。
    """
    errors = []
    for v in views:
        title = (v.get("title") or "")
        cls = title[:-len(".forward")] if title.endswith(".forward") else title
        if cls.split(".")[-1] in GENERIC_CLASSES:
            errors.append(f"图「{v['label']}」：{cls} 是通用基础层，不该单独成图")
    return errors


def build_fallback(views: list[dict]) -> str:
    """无脚本回退：每个视图一张节点表，保证脚本失效时页面仍可读。"""
    rows: list[str] = []
    for v in views:
        rows.append(f"<h2>{html.escape(v['label'])}：{html.escape(v['title'])}</h2>")
        for g in v.get("info", []):
            rows.append(f"<h3>{html.escape(g['title'])}</h3><table>")
            rows.extend(f"<tr><th>{html.escape(k)}</th><td>{html.escape(val)}</td></tr>"
                        for k, val in g["rows"])
            rows.append("</table>")
        rows.append("<table><tr><th>节点</th><th>形状 / 参数</th></tr>")
        for n in v["nodes"]:
            shape = n.get("shape") or n.get("detail") or ""
            rows.append("<tr><td>{}</td><td>{}</td></tr>".format(
                html.escape(n["name"]), html.escape(str(shape))))
            for sec in n.get("fold") or []:
                rows.append("<tr><td>{} · {}</td><td>{}</td></tr>".format(
                    html.escape(n["name"]), html.escape(sec.get("title") or ""),
                    html.escape(sec.get("text") or "")))
        rows.append("</table>")
    return "".join(rows)


def panel_cell(text: str) -> str:
    """面板单元格直接写进 HTML，所以转义后要把换行还原成 <br>。
    （节点和折叠的文字走另一条路：那里按 \\n 拆行，不需要 <br>。）"""
    return html.escape(str(text)).replace("\n", "<br>")


def build_config_panel(groups, note: list[str] | None = None) -> str:
    """把 (分组名, [(键, 值), ...]) 渲染成可折叠面板。note 是「前提」一节，逐条一行。"""
    blocks: list[str] = []
    if note:
        items = "".join(f"<li>{html.escape(s)}</li>" for s in note)
        blocks.append(f'<div class="cfg-note"><b>前提</b><ul>{items}</ul></div>')
    for title, rows in groups:
        body = "".join(
            f"<tr><th>{panel_cell(k)}</th><td>{panel_cell(v)}</td></tr>"
            for k, v in rows
        )
        blocks.append(
            f'<details class="cfg-group" open><summary>{html.escape(title)}</summary>'
            f"<table>{body}</table></details>"
        )
    return "".join(blocks)


def write_page(out: Path, meta: dict, views: list[dict],
               config_groups=None, config_note: list[str] | None = None,
               template: Path | None = None) -> Path:
    """把视图数据填进模板并写盘。meta 需要 title / description / summary / topics / tag。"""
    tpl = (template or TEMPLATE).read_text(encoding="utf-8")
    target = out.resolve().parent
    if REPO_ROOT in target.parents:
        libs = os.path.relpath(REPO_ROOT / "libs", target).replace(os.sep, "/")
        tpl = tpl.replace("../../libs/", f"{libs}/")

    tpl = tpl.replace("【首页摘要：纯文本，一句话说清这页讲什么】", meta["description"])
    tpl = tpl.replace("【首页摘要：可含 $...$ 公式；本篇的形状与超参逐条对准源码、config 与权重】",
                      meta["summary"])
    tpl = tpl.replace("【主题，逗号分隔，取自词表】", meta["topics"])
    tpl = tpl.replace("【标签】", meta["tag"])
    tpl = tpl.replace("【模型名】", meta["title"])

    tabs = "".join(
        '<button type="button" class="flow-tab{on}" data-view="{vid}">{label}</button>'.format(
            on=" on" if i == 0 else "", vid=v["id"], label=html.escape(v["label"]))
        for i, v in enumerate(views)
    )
    nav = f'<nav class="flow-bar">{tabs}</nav>'
    tpl = re.sub(r'<nav class="flow-bar">[\s\S]*?</nav>', lambda _: nav, tpl, count=1)

    fallback = f'<div class="flow-fallback">{build_fallback(views)}</div>\n  </noscript>'
    tpl = re.sub(r'<div class="flow-fallback">[\s\S]*?</div>\n  </noscript>', lambda _: fallback,
                 tpl, count=1)

    if config_groups:
        panel = (
            '<section class="flow-config" id="flow-config" hidden>'
            '<h2 class="cfg-title">config.json 参数</h2>'
            + build_config_panel(config_groups, config_note)
            + "</section>"
        )
        tpl = re.sub(r'<section class="flow-config"[\s\S]*?</section>', lambda _: panel, tpl, count=1)
    else:
        # 不需要参数面板时把按钮与面板整段删掉
        tpl = re.sub(r'\s*<button type="button" class="flow-cfg-btn"[\s\S]*?</button>', "", tpl, count=1)
        tpl = re.sub(r'\s*<section class="flow-config"[\s\S]*?</section>', "", tpl, count=1)

    tpl = tpl.replace('{"views":[【视图数组】]}',
                      json.dumps({"views": views}, ensure_ascii=False, separators=(",", ":")))

    leftover = re.findall(r"【[^】]*】", tpl)
    if leftover:
        raise SystemExit(f"占位符未替换完，模板改了？{leftover}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tpl, encoding="utf-8")
    return out


def build(plan: Path, out: Path | None = None) -> tuple[Path, dict]:
    """research/dataflow.md → 同一页面目录下的 index.html。"""
    if out is None:
        if plan.parent.name != "research":
            raise SystemExit(f"{plan} 不在 research/ 下，用 -o 指定输出路径")
        out = plan.parent.parent / "index.html"
    data = parse(plan)
    return write_page(out=out, meta=data["meta"], views=data["views"],
                      config_groups=data["config_groups"], config_note=data["config_note"]), data


def summary(path: Path, data: dict) -> str:
    views = data["views"]
    nodes = sum(len(v["nodes"]) for v in views)
    return f"已生成 {path}：{len(views)} 张图，{nodes} 个节点"


# ---------- 自检 ----------

TOY_PLAN = "\n".join([
    "# ToyModel",
    "",
    "- 描述：自检页。",
    "- 摘要：生成器自检，与任何真实模型无关。",
    "- 主题：模型结构",
    "- 标签：数据流",
    "",
    "## 资料",
    "",
    "- 源码：toy.py",
    "- config.json：自拟",
    "",
    "## 前提",
    "",
    "- 推理",
    "- 不传缓存",
    "",
    "## config.json",
    "",
    "| 组 | 名字 | 值 |",
    "|---|---|---|",
    "| 规模 | hidden_size | 16 |",
    "",
    "## ToyModel.forward（toy.py:1-9）",
    "",
    "### 节点",
    "",
    "| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |",
    "|---|---|---|---|---|---|",
    "| ids | 张量 | input_ids | int64 [B, T] | | toy.py:2 |",
    "| blk | 算子 | `2 × ToyBlock` | 16 → 16 | | toy.py:5 |",
    "| out | 张量 | hidden_states | fp32 [B, T, 16] | 第一行<br>第二行 | toy.py:6 |",
    "| kv | 缓存 | cache.update | 追加 | | toy.py:7 |",
    "",
    "### 连线",
    "",
    "- ids → blk",
    "- blk → out：hidden",
    "- kv->blk",
    "- blk 点开 → ToyBlock.forward",
    "",
    "### 折叠",
    "",
    "| id | 名字 | 内容 |",
    "|---|---|---|",
    "| blk | 公式 | $x + f(x)$，x 是输入 |",
    "",
    "### 面板",
    "",
    "| 组 | 名字 | 值 |",
    "|---|---|---|",
    "| 公式 | 残差 | $x + f(x)$ |",
    "| 公式 | 缩放 | $\\sqrt{d} \\odot x$ |",
    "| 公式 | 绝对值 | $|d|$ |",
    "| 配置 | hidden_size | 16 |",
    "",
    "## ToyBlock.forward（toy.py:10-20）",
    "",
    "### 节点",
    "",
    "| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |",
    "|---|---|---|---|---|---|",
    "| x | 张量 | x | fp32 [B, T, 16] | | toy.py:11 |",
])


def selftest() -> list[tuple[str, bool]]:
    """用一份与任何真实模型无关的计划跑一遍，确认生成器可用、该报错的地方会报错。"""
    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pdir = root / "wiki" / "toy" / "research"
        pdir.mkdir(parents=True)
        plan = pdir / "dataflow.md"
        plan.write_text(TOY_PLAN, encoding="utf-8")
        made, data = build(plan)
        t = made.read_text(encoding="utf-8")
        main_view = data["views"][0]
        checks += [
            ("输出到页面目录", made == root / "wiki" / "toy" / "index.html"),
            ("两张图", [v["id"] for v in data["views"]] == ["toymodel", "toyblock"]),
            ("点开指向第二张图", main_view["nodes"][1].get("drill") == "toyblock"),
            ("折叠挂在算子上", main_view["nodes"][1].get("fold") == [
                {"title": "公式", "text": "$x + f(x)$，x 是输入"}]),
            ("<br> 换行、反引号去掉", main_view["nodes"][2]["detail"] == "第一行\n第二行"
             and main_view["nodes"][1]["name"] == "2 × ToyBlock"),
            ("线上的字与无空格箭头", main_view["edges"][1].get("label") == "hidden"
             and main_view["edges"][2] == {"from": "kv", "to": "blk"}),
            ("面板按组读入", main_view["info"][0]["title"] == "公式"
             and main_view["info"][1]["rows"] == [["hidden_size", "16"]]),
            ("config 面板已渲染", "<summary>规模</summary>" in t),
            ("前提逐条写进 config 面板顶部", "<b>前提</b><ul><li>推理</li><li>不传缓存</li></ul>" in t),
            ("视图数据已内嵌", '"id":"toymodel"' in t and '"id":"toyblock"' in t),
            ("标签栏两个视图", t.count('class="flow-tab') == 2),
            ("无占位符残留", "【" not in t),
            ("无脚本回退含节点表", "<th>节点</th>" in t),
            ("资料和源码位置不进页面", "toy.py" not in t),
            ("公式里的反斜杠原样保留", "\\\\sqrt{d} \\\\odot x" in t and "\\sqrt{d} \\odot x" in t),
            ("公式里的竖线不拆列", "$|d|$" in t),
            ("引向 libs/dataflow 与 libs/shared", "libs/dataflow/dojo-flow.js" in t
             and "libs/shared/katex.min.js" in t),
        ]
        again, _ = build(plan)
        checks.append(("幂等：两次产出一致", again.read_text(encoding="utf-8") == t))

        for name, broken in [
            ("连线指向不存在的节点应当报错", TOY_PLAN.replace("- ids → blk", "- ids → nope")),
            ("点开的目标不存在应当报错", TOY_PLAN.replace("点开 → ToyBlock.forward", "点开 → Nope")),
            ("一个节点挂两条点开应当报错",
             TOY_PLAN.replace("- blk 点开 → ToyBlock.forward",
                              "- blk 点开 → ToyBlock.forward\n- blk 点开 → ToyModel.forward")),
            ("折叠写在张量上应当报错", TOY_PLAN.replace("| blk | 公式 |", "| ids | 公式 |")),
            ("折叠名字不对应当报错", TOY_PLAN.replace("| blk | 公式 |", "| blk | 残差 |")),
            ("类型写错应当报错", TOY_PLAN.replace("| ids | 张量 |", "| ids | 向量 |")),
            ("id 用了中文应当报错", TOY_PLAN.replace("| ids | 张量 |", "| 身份 | 张量 |")),
            ("缺少页面信息应当报错", TOY_PLAN.replace("- 摘要：生成器自检，与任何真实模型无关。\n", "")),
            ("缺少标签应当报错", TOY_PLAN.replace("- 标签：数据流\n", "")),
            ("标签不在词表应当报错", TOY_PLAN.replace("- 标签：数据流", "- 标签：不存在")),
            ("主题不在词表应当报错", TOY_PLAN.replace("- 主题：模型结构", "- 主题：不存在")),
            ("缺少前提应当报错", TOY_PLAN.replace("## 前提", "## 其它")),
            ("缺少资料应当报错", TOY_PLAN.replace("## 资料", "## 其它说明")),
            ("不认识的小节应当报错", TOY_PLAN.replace("### 面板", "### 备注")),
            ("算子第二行没写箭头应当报错", TOY_PLAN.replace("| 16 → 16 |", "| 16 |")),
            ("算子第二行写「同形状」应当报错",
             TOY_PLAN.replace("| 16 → 16 |", "| 16 → 同形状 |")),
            ("张量第二行留空应当报错", TOY_PLAN.replace("| int64 [B, T] | |", "| | |")),
            ("跨图同 id 指不同的量应当报错",
             TOY_PLAN.replace("| x | 张量 | x | fp32 [B, T, 16] | | toy.py:11 |",
                              "| ids | 张量 | other | fp32 [B, T, 16] | | toy.py:11 |")),
            ("散文里的下划线写成 \\_ 应当报错",
             TOY_PLAN.replace("| blk | 公式 | $x + f(x)$，x 是输入 |",
                              "| blk | 公式 | $x + f(x)$，x\\_是输入 |")),
            ("面板里的下划线写成 \\_ 也应当报错",
             TOY_PLAN.replace("| 配置 | hidden_size | 16 |",
                              "| 配置 | hidden_size | 见 hidden\\_size |")),
            ("config.json 表里的下划线写成 \\_ 也应当报错",
             TOY_PLAN.replace("| 规模 | hidden_size | 16 |",
                              "| 规模 | hidden_size | 见 hidden\\_size |")),
            ("前提里的下划线写成 \\_ 也应当报错",
             TOY_PLAN.replace("- 不传缓存", "- 不传\\_缓存")),
            ("页面信息的描述里下划线写成 \\_ 也应当报错",
             TOY_PLAN.replace("- 描述：自检页。", "- 描述：见 hidden\\_size。")),
            ("页面标题里的下划线写成 \\_ 也应当报错",
             TOY_PLAN.replace("# ToyModel", "# Toy\\_Model")),
            ("说明里写公式应当报错",
             TOY_PLAN.replace("| blk | 公式 | $x + f(x)$，x 是输入 |",
                              "| blk | 公式 | $x + f(x)$，x 是输入 |\n"
                              "| blk | 说明 | 再乘 $d^{-1/2}$ |")),
            ("算子第一行套了两层调用应当报错",
             TOY_PLAN.replace("| blk | 算子 | `2 × ToyBlock` |",
                              "| blk | 算子 | self.norm(self.proj(x)) |")),
            ("面板公式一条两个式子应当报错",
             TOY_PLAN.replace("| 公式 | 残差 | $x + f(x)$ |",
                              "| 公式 | 残差 | $x + f(x)$，再 $y = x$ |")),
            ("基础层单独成图应当报错",
             TOY_PLAN.replace("## ToyBlock.forward", "## RMSNorm.forward")
                     .replace("点开 → ToyBlock.forward", "点开 → RMSNorm.forward")),
            ("算子第三行非空应当报错",
             TOY_PLAN.replace("| blk | 算子 | `2 × ToyBlock` | 16 → 16 | |",
                              "| blk | 算子 | `2 × ToyBlock` | 16 → 16 | 塞了说明 |")),
        ]:
            plan.write_text(broken, encoding="utf-8")
            try:
                build(plan)
                checks.append((name, False))
            except SystemExit:
                checks.append((name, True))

        # 这条不该报错：说明里引一个符号（不是公式）
        ok_plan = TOY_PLAN.replace(
            "| blk | 公式 | $x + f(x)$，x 是输入 |",
            "| blk | 公式 | $x + f(x)$，x 是输入 |\n"
            "| blk | 说明 | 乘的是 $W$，见符号栏 |")
        plan.write_text(ok_plan, encoding="utf-8")
        try:
            build(plan)
            checks.append(("说明里引一个符号不算公式", True))
        except SystemExit:
            checks.append(("说明里引一个符号不算公式", False))
        plan.write_text(TOY_PLAN, encoding="utf-8")

        broken = root / "broken.html"
        broken.write_text("<html>【没人替换】</html>", encoding="utf-8")
        try:
            write_page(out=root / "x.html", meta={"title": "T", "description": "d", "summary": "s",
                                                 "topics": "模型结构", "tag": "数据流"},
                       views=[], template=broken)
            checks.append(("占位符残留时应当报错", False))
        except SystemExit:
            checks.append(("占位符残留时应当报错", True))

    return checks


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="读 research/dataflow.md，生成同目录上一级的 index.html")
    ap.add_argument("plan", nargs="?", type=Path, help="wiki/<name>/research/dataflow.md")
    ap.add_argument("-o", "--out", type=Path, help="输出路径，默认 wiki/<name>/index.html")
    ap.add_argument("--selftest", action="store_true", help="用虚构数据自检生成器")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(report(selftest(), "生成器"))
    if a.plan is None:
        ap.error("需要计划文件路径")
    if not a.plan.exists():
        raise SystemExit(f"计划文件不存在：{a.plan}")
    path, data = build(a.plan, a.out)
    print(summary(path, data))
