#!/usr/bin/env python3
"""数据流页的共享脚手架：把「视图数据」拼成完整 HTML。

为什么要有这个模块
------------------
画一张数据流图 = 内容 × 布局 × 呈现。这三件事的复用性完全不同：

  内容（哪些步骤、每格写什么、怎么连）  —— 每个模型独一无二，必须手写
  布局（节点摆哪、线怎么走）            —— 通用，交给 ELK
  呈现（骨架、交互、样式）              —— 通用，就是本模块 + dojo-flow.js

所以每个模型只需要写一份 JSON（views），其余全部由这里产出。
之前 Hy4 页面的生成脚本里，约 19% 是这类脚手架代码，每换一个模型都要抄一遍；
抽到这里之后，新模型只要：

    from dataflow_page import build_page
    build_page(out=Path("wiki/x-dataflow/index.html"), meta={...}, views=VIEWS)

数据结构（views 的契约）
------------------------
    view  = { id, label, title, nodes[], edges[], groups[]?, notes[]? }
    node  = { id, name, shape?, detail?, kind, drill? }
    edge  = { from, to, label? }

    kind 取值：
      tensor —— 张量，白底直角框（默认）
      op     —— 算子，浅蓝圆角框
      cache  —— 缓存，圆柱
      port   —— 跨视图传入/传出的量，虚线框

    命名约定（重要，见 verify_dataflow.py 的核查）：
      name 用**源码里被调用的符号**，不要用部署侧实现的名字。
      名字里出现的每个标识符都要能在权威源码里 grep 到。
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Iterable


PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="{description}">
  <meta name="dojo:summary" content="{summary}">
  <meta name="dojo:type" content="dataflow">
  <meta name="dojo:topics" content="{topics}">
  <meta name="dojo:tag" content="{tag}">
  <title>{title} · Dojo</title>
  <link rel="stylesheet" href="../../libs/katex.min.css">
  <script defer src="../../libs/katex.min.js"></script>
  <script defer src="../../libs/auto-render.min.js"
    onload="renderMathInElement(document.body, {{ delimiters: [
      {{left: '$$', right: '$$', display: true}},
      {{left: '$', right: '$', display: false}}
    ], throwOnError: false }});"></script>
  <link rel="stylesheet" href="../../libs/dojo-dataflow.css">
  <link rel="stylesheet" href="../../libs/dojo-flow.css">
  <noscript><style>
    /* 无脚本时画布不可交互，把节点表与参数面板直接显示出来 */
    .flow-config[hidden] {{ display: block !important; }}
    .flow-cfg-btn, .flow-zoom, .flow-bar {{ display: none; }}
  </style></noscript>
</head>
<body>
<div class="flow-app" id="flow-app">
  <noscript>
    <div class="flow-fallback">{fallback}</div>
  </noscript>
  <nav class="flow-bar">{tabs}</nav>
{cfg_button}  <div class="flow-zoom">
    <button type="button" data-act="out" title="缩小">−</button>
    <span class="flow-zoom-label" style="padding:0 6px;line-height:24px;font-size:12px;color:var(--flow-muted)">100%</span>
    <button type="button" data-act="in" title="放大">+</button>
    <button type="button" data-act="reset" title="复位">⟲</button>
  </div>
{cfg_panel}</div>
<script>window.DOJO_FLOW_DATA = {data};</script>
<script src="../../libs/elk.bundled.js"></script>
<script src="../../libs/dojo-flow.js"></script>
<script>
(function () {{
  var app = document.getElementById('flow-app');
  if (window.DojoFlow && window.DOJO_FLOW_DATA) {{
    window.__flow = window.DojoFlow.mount(app, window.DOJO_FLOW_DATA);
  }}
{cfg_script}}})();
</script>
</body>
</html>
"""

CFG_BUTTON = """  <button type="button" class="flow-cfg-btn" id="flow-cfg-btn"
          aria-expanded="false" aria-controls="flow-config">{label}</button>
"""

CFG_PANEL = """  <section class="flow-config" id="flow-config" hidden>
    <h2 class="cfg-title">{title}</h2>
    {body}
  </section>
"""

# 参数浮层的开关脚本。与视图切换解耦：点按钮或 Esc 关闭。
CFG_SCRIPT = """  var btn = document.getElementById('flow-cfg-btn');
  var panel = document.getElementById('flow-config');
  function setCfg(open) {
    if (!btn || !panel) return;
    panel.hidden = !open;
    btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    btn.classList.toggle('on', open);
  }
  if (btn && panel) {
    btn.addEventListener('click', function (e) {
      e.stopPropagation();
      setCfg(panel.hidden);
    });
    panel.addEventListener('click', function (e) { e.stopPropagation(); });
    window.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') setCfg(false);
    });
    if ((location.hash || '').replace('#', '') === 'config') setCfg(true);
  }
"""


def build_fallback(views: list[dict]) -> str:
    """无脚本回退：把每个视图渲染成一张表，保证脚本失效时页面仍可读。"""
    rows: list[str] = []
    for view in views:
        rows.append(f"<h2>{html.escape(view['label'])}：{html.escape(view['title'])}</h2>")
        rows.append("<table><tr><th>节点</th><th>形状 / 参数</th></tr>")
        for node in view["nodes"]:
            shape = node.get("shape") or node.get("detail") or ""
            rows.append(
                "<tr><td>{}</td><td>{}</td></tr>".format(
                    html.escape(node["name"]), html.escape(str(shape))
                )
            )
        rows.append("</table>")
    return "".join(rows)


def build_config_panel(groups: Iterable[tuple[str, list[tuple[str, str]]]],
                       title: str = "参数",
                       note: str = "") -> str:
    """把 (分组名, [(键, 值), ...]) 渲染成可折叠面板。"""
    blocks = []
    if note:
        blocks.append(f'<div class="cfg-note">{html.escape(note)}</div>')
    for group_title, rows in groups:
        body = "".join(
            f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"
            for k, v in rows
        )
        blocks.append(
            f'<details class="cfg-group" open><summary>{html.escape(group_title)}</summary>'
            f"<table>{body}</table></details>"
        )
    return "".join(blocks)


def build_page(
    out: Path,
    meta: dict[str, Any],
    views: list[dict],
    config_groups: Iterable[tuple[str, list[tuple[str, str]]]] | None = None,
    config_title: str = "参数",
    config_note: str = "",
    config_button: str = "参数",
) -> Path:
    """把视图数据拼成完整页面并写盘。

    meta 需要：title / description / summary / topics / tag
    """
    tabs = "".join(
        '<button type="button" class="flow-tab{on}" data-view="{vid}">{label}</button>'.format(
            on=" on" if i == 0 else "", vid=v["id"], label=html.escape(v["label"])
        )
        for i, v in enumerate(views)
    )
    cfg_button = cfg_panel = cfg_script = ""
    if config_groups:
        cfg_button = CFG_BUTTON.format(label=html.escape(config_button))
        cfg_panel = CFG_PANEL.format(
            title=html.escape(config_title),
            body=build_config_panel(config_groups, config_title, config_note),
        )
        cfg_script = CFG_SCRIPT

    page = PAGE.format(
        title=html.escape(meta["title"]),
        description=html.escape(meta["description"], quote=True),
        summary=html.escape(meta["summary"], quote=True),
        topics=html.escape(meta["topics"], quote=True),
        tag=html.escape(meta["tag"], quote=True),
        tabs=tabs,
        fallback=build_fallback(views),
        cfg_button=cfg_button,
        cfg_panel=cfg_panel,
        cfg_script=cfg_script,
        data=json.dumps({"views": views}, ensure_ascii=False, separators=(",", ":")),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out
