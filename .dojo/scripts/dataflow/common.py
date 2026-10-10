"""数据流两个脚本共用的常量与自检输出。

「点开」只连这个节点真的执行过的类；哪些类要成图，见 guides/dataflow/plan.md。
"""

from __future__ import annotations

import re

# 「不是图」的小节 → 它的 section 名。比较大小写不敏感（`## Config.json`
# 与 `## config.json` 都算）。这是唯一来源：build_page 按它分派、check_source
# 按它跳过，免得一处写 lower()、一处写精确匹配而漂开。
SKIP_HEAD_SECTIONS = {"资料": "sources", "前提": "scope", "config.json": "config"}
SKIP_HEADS = set(SKIP_HEAD_SECTIONS)


def op_shape_error(shape: str) -> str | None:
    """算子的第二行（形状箭头）写得对不对；对就返回 None，不对返回原因。

    必须恰好两段、都非空、都不是占位符。build_page（查计划）与 check_source
    （查页面）共用这一个判据，免得一处查两边、一处只查左边而结论不同。
    """
    shape = (shape or "").strip()
    if "→" not in shape and "->" not in shape:
        return "算子第二行要写「原来的形状 → 变完的形状」"
    sides = re.split(r"→|->", shape)
    if len(sides) != 2:
        return f"算子第二行只能有一个箭头，现在是「{shape}」"
    for side in sides:
        if not side.strip() or side.strip() in SHAPE_PLACEHOLDERS:
            return f"算子第二行两边都要写完整形状，现在是「{shape}」"
    return None


def skip_head_section(head: str) -> str | None:
    """`## <head>` 是非图小节就返回它的 section，是图就返回 None。"""
    return SKIP_HEAD_SECTIONS.get((head or "").strip().lower())


def is_skip_head(head: str) -> bool:
    """这个 `## <head>` 是不是「不是图」的小节（资料 / 前提 / config.json）。"""
    return skip_head_section(head) is not None
# 每张图下的四节
SUB_SECTIONS = ("节点", "连线", "折叠", "面板")
# 节点类型 → 页面里 kind 的取值
KIND = {"算子": "op", "张量": "tensor", "缓存": "cache"}
# 页面信息四行 → 页面里的名字
META_KEYS = {"描述": "description", "摘要": "summary", "主题": "topics", "标签": "tag"}
# 第二行里表示「形状没写全」的占位
SHAPE_PLACEHOLDERS = ("同形状", "同上", "不变", "—", "--")


def report(results: list[tuple[str, bool]], label: str) -> int:
    """打印自检结果，返回退出码。"""
    for name, ok in results:
        print(("  OK  " if ok else "  FAIL") + " " + name)
    failed = sum(not ok for _, ok in results)
    print(f"{label}自检失败：{failed} 项" if failed else f"{label}自检通过：{len(results)} 项")
    return 1 if failed else 0
