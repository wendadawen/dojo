"""数据流两个脚本共用的常量与自检输出。

「点开」只连这个节点真的执行过的类；哪些类要成图，见 guides/dataflow/plan.md。
"""

from __future__ import annotations

# 节点表里用来分隔小节的名字；这些小节不是图
SKIP_HEADS = {"资料", "前提", "config.json"}
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
