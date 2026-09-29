#!/usr/bin/env python3
"""数据流页的生成器：读计划文件 research/dataflow.md，生成页面。

用法（从仓库根运行）：
    python3 .dojo/scripts/dataflow-02-build.py wiki/<name>/research/dataflow.md

计划文件的格式见 guides/model-dataflow/plan.md。页面上的文字全部来自计划文件，
生成器只负责装配：读模板、替换占位符、把节点和面板写进页面。

它不做这些事：
  · 不定义页面骨架——骨架是 .dojo/templates/dataflow/index.html，唯一真相源
  · 不校验事实——那是审查和 dataflow-01-check.py 的事

数据结构（与引擎的契约，详见模板注释）
    view  = { id, label, title, nodes[], edges[], groups[]?, notes[]?, info[]? }
    info  = [{ title, rows: [[键, 值], ...] }]  说明面板，随视图切换；view(info=...) 传入
    node  = { id, name, shape?, detail?, kind, drill? }
    edge  = { from, to, label? }
    group = { label, members[], stroke? }     members 是本视图的节点 id，框在一起
    note  = { at, text, dx?, dy? }            at 是本视图的节点 id，批注画在它右侧
    kind：tensor | op | cache | port
    title 不显示在画布上，只用于无脚本表格的标题
"""

from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

# 模板相对仓库根的路径，从任意位置调用都能找到。
TEMPLATE = Path(__file__).resolve().parent.parent / "templates" / "dataflow" / "index.html"
REPO_ROOT = TEMPLATE.parents[3]


def node(nid: str, name: str, shape: str | None = None, kind: str = "op",
         src: str | None = None, detail: str | None = None,
         drill: str | None = None) -> dict:
    """构造一个节点。

    src 只是构建期的可读标注（源码行号或权重键），供人复核用，不写进页面。
    空值不写入结果，避免页面 JSON 里出现一堆 null。
    """
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
         info: Iterable[tuple[str, list[tuple[str, str]]]] | None = None) -> dict:
    """info 与 config_groups 同形：(分组名, [(键, 值)])，渲染成随视图切换的说明面板。"""
    out = {
        "id": vid,
        "label": label,
        "title": title,
        "nodes": nodes,
        "edges": edges,
        "groups": groups or [],
        "notes": notes or [],
    }
    if info:
        out["info"] = [{"title": t, "rows": [[k, v] for k, v in rows]} for t, rows in info]
    return out


def build_fallback(views: Iterable[dict]) -> str:
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
        rows.append("</table>")
    return "".join(rows)


def build_config_panel(groups: Iterable[tuple[str, list[tuple[str, str]]]],
                       note: str = "") -> str:
    """把 (分组名, [(键, 值), ...]) 渲染成可折叠面板。"""
    blocks: list[str] = []
    if note:
        blocks.append(f'<div class="cfg-note">{html.escape(note)}</div>')
    for title, rows in groups:
        body = "".join(
            f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"
            for k, v in rows
        )
        blocks.append(
            f'<details class="cfg-group" open><summary>{html.escape(title)}</summary>'
            f"<table>{body}</table></details>"
        )
    return "".join(blocks)


def page(out: Path, meta: dict[str, Any], views: list[dict],
         config_groups: Iterable[tuple[str, list[tuple[str, str]]]] | None = None,
         config_title: str = "config.json 参数",
         config_note: str = "",
         config_button: str = "config.json",
         template: Path | None = None) -> Path:
    """把视图数据填进模板并写盘。

    meta 需要：title / description / summary / topics / tag
    """
    tpl = (template or TEMPLATE).read_text(encoding="utf-8")
    # 模板按 wiki/<name>/ 写 ../../libs/；页面目录更深或更浅时按实际层级改写
    target = out.resolve().parent
    if REPO_ROOT in target.parents:
        libs = os.path.relpath(REPO_ROOT / "libs", target).replace(os.sep, "/")
        tpl = tpl.replace("../../libs/", f"{libs}/")

    tpl = tpl.replace("【首页摘要：纯文本，一句话说清这页讲什么】", meta["description"])
    tpl = tpl.replace("【首页摘要：可含 $...$ 公式；本篇的形状与超参逐条对准源码、config 与权重】",
                      meta["summary"])
    tpl = tpl.replace("【主题，逗号分隔，取自固定大类】", meta["topics"])
    tpl = tpl.replace("【模型名】", meta["title"])

    tabs = "".join(
        '<button type="button" class="flow-tab{on}" data-view="{vid}">{label}</button>'.format(
            on=" on" if i == 0 else "", vid=v["id"], label=html.escape(v["label"]))
        for i, v in enumerate(views)
    )
    # 替换内容里可能有公式的反斜杠，统一用函数形式传入，不让 re 当成转义
    nav = f'<nav class="flow-bar">{tabs}</nav>'
    tpl = re.sub(r'<nav class="flow-bar">[\s\S]*?</nav>', lambda _: nav, tpl, count=1)

    fallback = f'<div class="flow-fallback">{build_fallback(views)}</div>\n  </noscript>'
    tpl = re.sub(r'<div class="flow-fallback">[\s\S]*?</div>\n  </noscript>', lambda _: fallback,
                 tpl, count=1)

    if config_groups:
        panel = (
            '<section class="flow-config" id="flow-config" hidden>'
            f'<h2 class="cfg-title">{html.escape(config_title)}</h2>'
            + build_config_panel(config_groups, config_note)
            + "</section>"
        )
        tpl = re.sub(r'<section class="flow-config"[\s\S]*?</section>', lambda _: panel, tpl, count=1)
        tpl = tpl.replace('config.json</button>', f'{html.escape(config_button)}</button>', 1)
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


KIND = {"算子": "op", "张量": "tensor", "缓存": "cache"}
DRILL_RE = re.compile(r"^-\s*(\S+)\s+点开\s*(?:→|->)\s*(.+?)\s*$")
EDGE_RE = re.compile(r"^-\s*([^\s→]+?)\s*(?:→|->)\s*([^\s：:]+)\s*(?:[：:]\s*(.+?))?\s*$")
META_KEYS = {"描述": "description", "摘要": "summary", "主题": "topics"}


def _cells(line: str) -> list[str]:
    """一行 Markdown 表格拆成单元格：\\| 是字面竖线，<br> 换行，反引号去掉。

    $...$ 里的竖线是公式，不拆列。
    """
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


def _table(rows: list[tuple[int, str]], need: list[str], where: str) -> list[tuple[int, dict]]:
    """表头按列名取值；缺列时报出表头所在行。"""
    if not rows:
        return []
    head_no, head = rows[0]
    names = _cells(head)
    missing = [n for n in need if n not in names]
    if missing:
        raise SystemExit(f"第 {head_no} 行：{where}的表头缺少列 {'、'.join(missing)}")
    out = []
    for no, line in rows[1:]:
        cells = _cells(line)
        if all(re.fullmatch(r":?-{3,}:?", c) for c in cells if c):
            continue
        out.append((no, dict(zip(names, cells + [""] * (len(names) - len(cells))))))
    return out


def _grouped(rows: list[tuple[int, dict]]) -> list[tuple[str, list[tuple[str, str]]]]:
    groups: dict[str, list[tuple[str, str]]] = {}
    for _, r in rows:
        groups.setdefault(r["组"], []).append((r["名字"], r["值"]))
    return list(groups.items())


def parse_dataflow_md(text: str) -> dict:
    """按 guides/model-dataflow/plan.md 的格式读计划文件，返回 page() 需要的参数。"""
    meta: dict[str, str] = {"tag": "数据流"}
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
                                  "drills": [], "panel": []})
        elif s.startswith("### "):
            sub = s[4:].strip()
            if section != "view" or sub not in ("节点", "连线", "面板"):
                raise SystemExit(f"第 {no} 行：不认识的小节「{sub}」，只能是 节点 / 连线 / 面板")
        elif section is None and s.startswith("-"):
            key, _, value = s[1:].strip().partition("：")
            if not _:
                key, _, value = s[1:].strip().partition(":")
            if key.strip() in META_KEYS:
                meta[META_KEYS[key.strip()]] = value.strip()
        elif section == "sources" and s.startswith("-"):
            sources.append(s[1:].strip().replace("`", ""))
        elif section == "scope" and s.startswith("-"):
            scope.append(s[1:].strip().replace("`", ""))
        elif section == "config" and s.startswith("|"):
            config_rows.append((no, s))
        elif section == "view" and sub in ("节点", "面板") and s.startswith("|"):
            views_raw[-1]["nodes" if sub == "节点" else "panel"].append((no, s))
        elif section == "view" and sub == "连线" and s.startswith("-"):
            m = DRILL_RE.match(s)
            if m:
                views_raw[-1]["drills"].append((no, m.group(1), m.group(2)))
                continue
            m = EDGE_RE.match(s)
            if not m:
                raise SystemExit(f"第 {no} 行：连线要写成「- a → b」或「- a → b：线上的字」")
            views_raw[-1]["edges"].append((no, m.group(1), m.group(2), m.group(3)))

    for key in ("title", "description", "summary", "topics"):
        if key not in meta:
            name = "模型名（# 标题）" if key == "title" else next(k for k, v in META_KEYS.items() if v == key)
            raise SystemExit(f"计划文件缺少页面信息：{name}")
    if not views_raw:
        raise SystemExit("计划文件里没有图（## <类名>.forward（...））")

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
        for no, r in _table(v["nodes"], ["id", "类型", "第一行", "第二行", "第三行"], where + "节点表"):
            nid = r["id"]
            if not nid or nid in ids:
                raise SystemExit(f"第 {no} 行：id「{nid}」为空或在{where}里重复")
            if r["类型"] not in KIND:
                raise SystemExit(f"第 {no} 行：类型「{r['类型']}」只能是 算子 / 张量 / 缓存")
            ids.add(nid)
            nodes.append(node(nid, r["第一行"], r["第二行"] or None, KIND[r["类型"]],
                              src=r.get("源码") or None, detail=r["第三行"] or None))
        if not nodes:
            raise SystemExit(f"第 {v['no']} 行：{where}没有节点表")
        by_id = {n["id"]: n for n in nodes}
        edges = []
        for no, a, b, label in v["edges"]:
            for end in (a, b):
                if end not in by_id:
                    raise SystemExit(f"第 {no} 行：{where}里没有节点「{end}」")
            edges.append(edge(a, b, label))
        for no, nid, target in v["drills"]:
            if nid not in by_id:
                raise SystemExit(f"第 {no} 行：{where}里没有节点「{nid}」")
            tid = find_view(target)
            if tid is None:
                raise SystemExit(f"第 {no} 行：点开的目标「{target}」不是任何一张图的标题")
            by_id[nid]["drill"] = tid
        info = _grouped(_table(v["panel"], ["组", "名字", "值"], where + "面板"))
        title = re.split(r"[（(]", v["head"], maxsplit=1)[0].strip()
        views.append(view(v["id"], v["label"], title, nodes, edges, info=info))

    config = _grouped(_table(config_rows, ["组", "名字", "值"], "config.json 表"))
    if not sources:
        raise SystemExit("计划文件缺少「资料」一节")
    if not scope:
        raise SystemExit("计划文件缺少「前提」一节")
    return {"meta": meta, "views": views, "config_groups": config or None,
            "config_note": "前提：" + "；".join(scope)}


def build_from_md(md: Path, out: Path | None = None) -> Path:
    """research/dataflow.md → 同一页面目录下的 index.html。"""
    if out is None:
        if md.parent.name != "research":
            raise SystemExit(f"{md} 不在 research/ 下，用 -o 指定输出路径")
        out = md.parent.parent / "index.html"
    args = parse_dataflow_md(md.read_text(encoding="utf-8"))
    return page(out=out, **args)


def _selftest() -> int:
    """自检：用一套与任何真实模型无关的数据跑一遍，确认生成器可用。

    为什么要有：生成器只在「写新页面」时被用到，平时没有调用方。没有自检
    的话，它坏了也不会有人发现，直到下次做新页面才暴露。
    """
    import tempfile

    views = [
        view(
            "main", "ToyModel", "自检：主干",
            nodes=[
                node("ids", "input_ids", "int64 [B, T]", "tensor"),
                node("emb", "embed_tokens", "32 → 16", "op", detail="toy [32, 16]"),
                node("h", "hidden_states", "fp32 [B, T, 16]", "tensor"),
                node("blk", "2 × ToyBlock", "点开看单层", "op", drill="block"),
                node("out", "logits", "fp32 [B, T, 32]", "tensor"),
                node("kv", "cache", "推理时复用", "cache"),
                node("ext", "external_port", "跨视图传入", "port"),
            ],
            edges=[
                edge("ids", "emb"), edge("emb", "h"), edge("h", "blk"),
                edge("blk", "out"), edge("kv", "blk", "past"), edge("ext", "blk"),
            ],
            groups=[{"label": "主干",
                     "members": ["ids", "emb", "h", "blk", "out"], "stroke": "#9db6d8"}],
            info=[("这张图", [("做什么", "自检用的玩具主干")])],
        ),
        view(
            "block", "ToyBlock", "自检：单层",
            nodes=[
                node("x", "x", "fp32 [B, T, 16]", "tensor"),
                node("ln", "norm", "[16]", "op"),
                node("y", "y", "fp32 [B, T, 16]", "tensor"),
                node("add", "x + y", "残差", "op"),
                node("z", "out", "fp32 [B, T, 16]", "tensor"),
            ],
            edges=[edge("x", "ln"), edge("ln", "y"), edge("x", "add"),
                   edge("y", "add"), edge("add", "z")],
        ),
    ]
    meta = {"title": "ToyModel", "description": "自检页。",
            "summary": "生成器自检，与任何真实模型无关。",
            "topics": "模型结构", "tag": "数据流"}

    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory() as tmp:
        # 1) 带参数面板：按钮与面板都在，占位符无残留
        out1 = Path(tmp) / "with-cfg.html"
        page(out=out1, meta=meta, views=views,
             config_groups=[("规模", [("hidden_size", "16")])],
             config_note="自检")
        t1 = out1.read_text(encoding="utf-8")
        checks += [
            ("带面板：按钮元素存在", '<button type="button" class="flow-cfg-btn"' in t1),
            ("带面板：面板元素存在", 'id="flow-config"' in t1),
            ("带面板：cfg 分组已渲染", '<summary>规模</summary>' in t1),
            ("带面板：无占位符残留", "【" not in t1),
            ("带面板：视图数据已内嵌", '"id":"main"' in t1 and '"id":"block"' in t1),
            ("带面板：标签栏含两个视图", t1.count('class="flow-tab') == 2),
            ("带面板：下钻标记保留", '"drill":"block"' in t1),
            ("说明：视图 info 已内嵌", '"info":[{"title":"这张图"' in t1),
            ("说明：无脚本回退含说明", "<h3>这张图</h3>" in t1),
            ("说明：按钮与面板存在", 'id="flow-info-btn"' in t1 and 'id="flow-info"' in t1),
        ]

        # 2) 不带参数面板：按钮与面板元素都应消失
        out2 = Path(tmp) / "no-cfg.html"
        page(out=out2, meta=meta, views=views)
        t2 = out2.read_text(encoding="utf-8")
        checks += [
            ("无面板：按钮元素已删", '<button type="button" class="flow-cfg-btn"' not in t2),
            ("无面板：面板元素已删", 'id="flow-config"' not in t2),
            ("无面板：无占位符残留", "【" not in t2),
        ]

        # 3) 幂等：同一输入两次产出相同
        out3 = Path(tmp) / "again.html"
        page(out=out3, meta=meta, views=views)
        checks.append(("幂等：两次产出一致", out2.read_text(encoding="utf-8") == t2))

        # 4) 未替换的占位符应当报错，而不是静默产出
        broken = Path(tmp) / "broken.html"
        broken.write_text("<html>【没人替换】</html>", encoding="utf-8")
        try:
            page(out=Path(tmp) / "x.html", meta=meta, views=views, template=broken)
            checks.append(("占位符残留时应当报错", False))
        except SystemExit:
            checks.append(("占位符残留时应当报错", True))

        # 5) 从计划文件生成：格式照 guides/model-dataflow/plan.md
        plan = "\n".join([
            "# ToyModel",
            "",
            "- 描述：自检页。",
            "- 摘要：生成器自检，与任何真实模型无关。",
            "- 主题：模型结构",
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
            "| blk | 算子 | `2 × ToyBlock` | 16 → 16 | 第一行<br>第二行 | toy.py:5 |",
            "| out | 张量 | hidden_states | fp32 [B, T, 16] | | toy.py:6 |",
            "| kv | 缓存 | cache.update | 追加 | | toy.py:7 |",
            "",
            "### 连线",
            "",
            "- ids → blk",
            "- blk → out：hidden",
            "- kv->blk",
            "- blk 点开 → ToyBlock.forward",
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
        pdir = Path(tmp) / "toy" / "research"
        pdir.mkdir(parents=True)
        (pdir / "dataflow.md").write_text(plan, encoding="utf-8")
        made = build_from_md(pdir / "dataflow.md")
        data = parse_dataflow_md(plan)
        main_view = data["views"][0]
        t5 = made.read_text(encoding="utf-8")
        checks += [
            ("计划文件：输出到页面目录", made == Path(tmp) / "toy" / "index.html"),
            ("计划文件：两张图", [v["id"] for v in data["views"]] == ["toymodel", "toyblock"]),
            ("计划文件：点开指向第二张图", main_view["nodes"][1].get("drill") == "toyblock"),
            ("计划文件：<br> 换行、反引号去掉", main_view["nodes"][1]["detail"] == "第一行\n第二行"
             and main_view["nodes"][1]["name"] == "2 × ToyBlock"),
            ("计划文件：线上的字与无空格箭头", main_view["edges"][1].get("label") == "hidden"
             and main_view["edges"][2] == {"from": "kv", "to": "blk"}),
            ("计划文件：面板按组读入", main_view["info"][0]["title"] == "公式"
             and main_view["info"][1]["rows"] == [["hidden_size", "16"]]),
            ("计划文件：config 面板", "<summary>规模</summary>" in t5),
            ("计划文件：前提写进 config 面板顶部", "前提：推理；不传缓存" in t5),
            ("计划文件：资料和标题里的源码位置不进页面", "toy.py" not in t5),
            ("计划文件：公式里的反斜杠原样保留", "\\\\sqrt{d} \\\\odot x" in t5 and "\\sqrt{d} \\odot x" in t5),
            ("计划文件：公式里的竖线不拆列", "$|d|$" in t5),
            ("计划文件：节点的源码位置不进页面", "toy.py:5" not in t5 and "toy.py:2" not in t5),
        ]
        for name, broken_plan in [
            ("连线指向不存在的节点应当报错", plan.replace("- ids → blk", "- ids → nope")),
            ("点开的目标不存在应当报错", plan.replace("点开 → ToyBlock.forward", "点开 → Nope")),
            ("类型写错应当报错", plan.replace("| ids | 张量 |", "| ids | 向量 |")),
            ("缺少页面信息应当报错", plan.replace("- 摘要：生成器自检，与任何真实模型无关。\n", "")),
            ("缺少前提应当报错", plan.replace("## 前提", "## 其它").replace("- 推理\n", "")
             .replace("- 不传缓存\n", "")),
            ("缺少资料应当报错", plan.replace("## 资料", "## 其它说明").replace("- 源码：toy.py\n", "")
             .replace("- config.json：自拟\n", "")),
        ]:
            try:
                parse_dataflow_md(broken_plan)
                checks.append((name, False))
            except SystemExit:
                checks.append((name, True))

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

    ap = argparse.ArgumentParser(description="读 research/dataflow.md，生成同目录上一级的 index.html")
    ap.add_argument("plan", nargs="?", type=Path, help="wiki/<name>/research/dataflow.md")
    ap.add_argument("-o", "--out", type=Path, help="输出路径，默认 wiki/<name>/index.html")
    ap.add_argument("--selftest", action="store_true", help="用虚构数据自检生成器")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(_selftest())
    if a.plan is None:
        ap.error("需要计划文件路径")
    if not a.plan.exists():
        raise SystemExit(f"计划文件不存在：{a.plan}")
    result = build_from_md(a.plan, a.out)
    data = parse_dataflow_md(a.plan.read_text(encoding="utf-8"))
    count = sum(len(v["nodes"]) for v in data["views"])
    print(f"已生成 {result}：{len(data['views'])} 张图，{count} 个节点")
