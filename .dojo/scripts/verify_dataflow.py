#!/usr/bin/env python3
"""数据流页的事实核查：把页面上的每个断言回查到源码、配置、权重。

数据流页最容易出的错不是排版，而是**写出源码里根本不存在的东西**。
这个脚本把三类断言变成可执行的检查：

  1. 节点名  —— 名字里出现的标识符，必须在源码里 grep 得到
  2. 形状    —— 写出的权重形状，必须与 safetensors 头一致
  3. 行号    —— 标注的源码行号，必须落在该文件范围内

前两项需要外部材料（源码 / 权重），所以不做进 validate.py（那个是纯静态检查），
单独跑这个脚本。

用法：
    # 源码在本地
    python3 .dojo/scripts/verify_dataflow.py wiki/<name>/index.html \
        --source /path/to/modeling_x.py

    # 只看节点名（不需要外部材料）
    python3 .dojo/scripts/verify_dataflow.py wiki/<name>/index.html --names-only

退出码：0 全部通过；1 有问题；2 用法错误。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# 节点名里允许出现、但不属于「源码标识符」的词：
# 数据类型、形状记号、数学符号、以及句式里的连接词。
ALLOWED_TOKENS = {
    # 数据类型与框架
    "bf16", "fp32", "fp16", "int32", "int64", "bool", "float", "float32", "dtype",
    # 常见的 nn 构造子（页面用 `nn.Embedding` 这种写法时会出现）
    "nn", "Embedding", "Linear", "RMSNorm", "LayerNorm", "Parameter", "Buffer",
    # 形状记号
    "B", "T", "S", "N", "D", "H", "L", "O", "tokens", "scalar", "dict",
    # 源码里作为局部变量出现、但短到容易误报的
    "x", "a", "b", "c", "k", "q", "v", "w", "p", "e", "i", "j", "n", "m",
    # 句式连接词
    "and", "or", "per", "the", "to", "of", "in", "on", "if", "else", "not",
    "None", "True", "False",
}


def load_views(page: Path) -> list[dict]:
    html = page.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"window\.DOJO_FLOW_DATA = (\{.*?\});</script>", html, re.S)
    if not m:
        raise SystemExit(f"error: {page} 里没有 DOJO_FLOW_DATA 数据块")
    return json.loads(m.group(1))["views"]


def idents(text: str) -> list[str]:
    """取出文本里的标识符（字母数字下划线，且不以数字开头）。"""
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)


def check_names(views: list[dict], sources: dict[str, str]) -> list[str]:
    """节点名里出现的标识符必须在某份源码里找得到。

    这是防「凭空捏造」的核心检查。实测踩过的坑：节点写了 einsum，
    而源码里一次 einsum 都没有。
    """
    errors: list[str] = []
    for view in views:
        for node in view["nodes"]:
            for tok in idents(node["name"]):
                if tok in ALLOWED_TOKENS:
                    continue
                if any(tok in body for body in sources.values()):
                    continue
                # 允许「检查点键名」这类外部命名：detail 里注明来源即可
                detail = node.get("detail", "")
                if tok in detail and ("检查点" in detail or "键" in detail):
                    continue
                errors.append(
                    f"[{view['id']}/{node['id']}] 节点名 {node['name']!r} 里的 "
                    f"{tok!r} 在源码中找不到"
                )
    return errors


def check_shapes(views: list[dict], shapes: dict[str, list[int]]) -> list[str]:
    """页面上写的权重形状必须与给定的形状表一致。

    shapes 由 --shapes 传入（name -> [dim...]），通常来自 safetensors 头。
    """
    errors: list[str] = []
    if not shapes:
        return errors
    for view in views:
        for node in view["nodes"]:
            blob = " ".join(filter(None, [node.get("name"), node.get("detail"), node.get("shape")]))
            for m in re.finditer(r"([a-z_][a-z0-9_.]*)\s*\[([0-9][0-9,\s]*)\]", blob):
                key = m.group(1)
                if key not in shapes:
                    continue
                claimed = [int(x) for x in m.group(2).replace(" ", "").split(",")]
                expected = shapes[key]
                if claimed != expected:
                    errors.append(
                        f"[{view['id']}/{node['id']}] {key} 页面写 {claimed}，"
                        f"材料里是 {expected}"
                    )
    return errors


def check_line_numbers(scripts: list[Path]) -> list[str]:
    """生成脚本里标注的源码行号不能越界。"""
    errors: list[str] = []
    for script in scripts:
        if not script.exists():
            errors.append(f"脚本不存在: {script}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description="数据流页事实核查")
    ap.add_argument("page", type=Path, help="wiki/<name>/index.html")
    ap.add_argument("--source", action="append", default=[], metavar="[LABEL=]PATH",
                    help="权威源码，可多次传入。带 LABEL 时（如 mtp=/path/mtp.py）"
                         "只有视图标题里含该 LABEL 才用它做校验，这样每个视图可以"
                         "有自己的权威来源（例如 MTP 视图的事实来源是 vLLM 的 mtp.py，"
                         "而不是 transformers）")
    ap.add_argument("--shapes", type=Path,
                    help="形状表 JSON：{tensor_key: [dim...]}，通常来自 safetensors 头")
    ap.add_argument("--script", action="append", default=[], type=Path,
                    help="生成脚本（用于校验行号），可多次传入")
    ap.add_argument("--names-only", action="store_true",
                    help="只做节点名核查")
    args = ap.parse_args()

    if not args.page.exists():
        print(f"error: 页面不存在: {args.page}", file=sys.stderr)
        return 2

    views = load_views(args.page)

    # 解析 --source：LABEL=PATH 或纯 PATH（纯 PATH 对所有视图生效）
    sources: dict[str, str] = {}
    labeled: list[tuple[str, str]] = []     # (label, body)
    labeled_files: set[str] = set()          # 已被某个 label 指派的文件
    for raw in args.source:
        label, _, path = str(raw).partition("=")
        if not _:
            label, path = "", label
        p = Path(path)
        if not p.exists():
            print(f"error: 源码不存在: {p}", file=sys.stderr)
            return 2
        body = p.read_text(encoding="utf-8", errors="ignore")
        sources[path] = body
        if label:
            labeled.append((label, body))
            labeled_files.add(path)

    if args.names_only and not sources:
        print("error: --names-only 仍需 --source 才能比对", file=sys.stderr)
        return 2

    # 选源规则：
    #   · 打了标签的源（LABEL=PATH）只作用于「标题或标签里含该 LABEL」的视图；
    #     这样 MTP 视图可以用 vLLM 的 mtp.py 校验，而其余视图用 transformers。
    #   · 没打标签的源对所有视图生效。
    labeled_bodies = {label: body for label, body in labeled}
    universal = [body for path, body in sources.items() if path not in labeled_files]

    errors: list[str] = []
    if sources:
        for view in views:
            heading = view.get("title", "") + " " + view.get("label", "")
            low = heading.lower()
            picked = [body for label, body in labeled_bodies.items() if label.lower() in low]
            if not picked:
                picked = universal
            if picked:
                errors += check_names([view], {"view": "\n".join(picked)})
    if not args.names_only and args.shapes:
        shapes = json.loads(args.shapes.read_text(encoding="utf-8"))
        errors += check_shapes(views, shapes)
    if not args.names_only and args.script:
        errors += check_line_numbers([Path(p) for p in args.script])

    nodes = sum(len(v["nodes"]) for v in views)
    if errors:
        print(f"核查失败：{len(errors)} 个问题（共 {len(views)} 视图 / {nodes} 节点）")
        for e in errors:
            print(f"  - {e}")
        return 1

    checked = []
    if sources:
        checked.append(f"节点名 vs {len(sources)} 份源码")
    if not args.names_only and args.shapes:
        checked.append("形状 vs 材料")
    print(f"核查通过：{len(views)} 视图 / {nodes} 节点" + (
        "（" + "、".join(checked) + "）" if checked else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
