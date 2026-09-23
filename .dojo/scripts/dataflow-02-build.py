#!/usr/bin/env python3
"""数据流页的生成器：把一份纯数据脚本变成页面。

为什么要有这个脚本
------------------
画数据流页分两层：

  数据（哪些视图、哪些节点、每条边、config 分组）—— 每个模型独一无二
  装配（读模板、替换占位符、把 dict 渲染成 HTML 片段）—— 与模型无关

没有这层时，每做一页都要把装配代码重写一遍，里面一个模型相关的字都没有。
hy4 与 qwen3 两页各写过一次 node/edge/build_fallback/build_config_panel，
合计约 100 行重复。这个脚本把这层收拢到一处。

它不做这些事：
  · 不定义页面骨架——骨架是 .dojo/templates/dataflow/index.html，唯一真相源
  · 不生成数据——数据由各页自己的数据脚本提供
  · 不校验事实——那是 dataflow-01-check.py 的事

数据脚本怎么写
--------------
    import importlib, sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".dojo/scripts"))
    build = importlib.import_module("dataflow-02-build")   # 模块名带连字符

    VIEWS = [build.view("main", "Qwen3Model", "主干：...", nodes=[...], edges=[...])]
    build.page(
        out=Path("wiki/<name>/index.html"),
        meta={"title": ..., "description": ..., "summary": ...,
              "topics": "模型结构", "tag": "数据流"},
        views=VIEWS,
        config_groups=[("分组名", [("键", "值")])],
        config_note="config.json 原值，共 N 个键",
    )

数据结构（与引擎的契约，详见模板注释）
    view = { id, label, title, nodes[], edges[], groups[]?, notes[]? }
    node = { id, name, shape?, detail?, kind, drill? }
    edge = { from, to, label? }
    kind：tensor | op | cache | port
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Iterable

# 模板相对仓库根的路径。数据脚本从任意位置调用都能找到。
TEMPLATE = Path(__file__).resolve().parent.parent / "templates" / "dataflow" / "index.html"


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
         groups: list[dict] | None = None, notes: list[dict] | None = None) -> dict:
    return {
        "id": vid,
        "label": label,
        "title": title,
        "nodes": nodes,
        "edges": edges,
        "groups": groups or [],
        "notes": notes or [],
    }


def build_fallback(views: Iterable[dict]) -> str:
    """无脚本回退：每个视图一张节点表，保证脚本失效时页面仍可读。"""
    rows: list[str] = []
    for v in views:
        rows.append(f"<h2>{html.escape(v['label'])}：{html.escape(v['title'])}</h2>")
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
    tpl = re.sub(r'<nav class="flow-bar">[\s\S]*?</nav>',
                 f'<nav class="flow-bar">{tabs}</nav>', tpl, count=1)

    tpl = re.sub(r'<div class="flow-fallback">[\s\S]*?</div>\n  </noscript>',
                 f'<div class="flow-fallback">{build_fallback(views)}</div>\n  </noscript>',
                 tpl, count=1)

    if config_groups:
        panel = (
            '<section class="flow-config" id="flow-config" hidden>'
            f'<h2 class="cfg-title">{html.escape(config_title)}</h2>'
            + build_config_panel(config_groups, config_note)
            + "</section>"
        )
        tpl = re.sub(r'<section class="flow-config"[\s\S]*?</section>', panel, tpl, count=1)
        tpl = tpl.replace('config.json</button>', f'{html.escape(config_button)}</button>', 1)
    else:
        # 不需要参数面板时把按钮与面板整段删掉
        tpl = re.sub(r'\s*<button type="button" class="flow-cfg-btn"[\s\S]*?</button>', "", tpl, count=1)
        tpl = re.sub(r'\s*<section class="flow-config"[\s\S]*?</section>', "", tpl, count=1)
        tpl = re.sub(r'\s*// 参数面板开关：[\s\S]*?\n  }\n(?=})', "", tpl, count=1)

    tpl = tpl.replace('{"views":[【视图数组】]}',
                      json.dumps({"views": views}, ensure_ascii=False, separators=(",", ":")))

    leftover = re.findall(r"【[^】]*】", tpl)
    if leftover:
        raise SystemExit(f"占位符未替换完，模板改了？{leftover}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tpl, encoding="utf-8")
    return out


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

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(("  OK  " if ok else "  FAIL") + " " + name)
    if failed:
        print(f"生成器自检失败：{len(failed)} 项")
        return 1
    print(f"生成器自检通过：{len(checks)} 项")
    return 0


if __name__ == "__main__":
    import sys as _sys

    if "--selftest" in _sys.argv:
        raise SystemExit(_selftest())
    raise SystemExit(
        "这是生成器模块，不单独执行。用法见文件头注释；\n"
        "数据脚本通过 importlib 加载它。自检：--selftest"
    )
