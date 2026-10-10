#!/usr/bin/env python3
"""数据流页的核查：事实对不对，以及连线画得干不干净。

数据流页最容易出的错不是排版，而是写出源码里不存在的东西。这个脚本把断言变成可执行的检查：

  1. 源码位置 —— 传入计划文件时，「源码」列指的行不是空行或注释，且含节点名
  2. 节点名   —— 名字里出现的标识符必须在源码里出现；短词表里的词跳过
  3. 形状     —— 写出的权重形状，必须与 safetensors 头一致
  4. 写法     —— 形状箭头两边都要有；折叠的名字只能是「公式」「符号」或「说明」；算子没有公式时给出提示

前三项需要外部材料（源码 / 权重），所以不做进 site/check_page.py（那是纯静态检查），单独跑这个脚本。

用法（从仓库根运行）：
    # 计划文件的源码位置
    python3 .dojo/scripts/dataflow/check_source.py wiki/<name>/research/dataflow.md \
        --source-root /path/to/transformers

    # 节点名与权重形状
    python3 .dojo/scripts/dataflow/check_source.py wiki/<name>/index.html \
        --source /path/to/modeling_x.py --shapes wiki/<name>/research/sources/shapes.json

退出码：0 全部通过；1 有问题；2 用法错误。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import SKIP_HEADS, report  # noqa: E402
from build_page import split_cells  # noqa: E402  （两个脚本用同一套拆列）

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
# 节点第一行照源码里的名字写。这里只给仍可能出现的短名留对照。
NAME_ALIASES = {
    "x": ("hidden_states", "inputs_embeds", "last_hidden_state"),
    "q": ("query_states", "query"),
    "k": ("key_states", "key"),
    "v": ("value_states", "value"),
}


# 出现在方括号左边的数据类型/记号，不是权重键名
DTYPE_WORDS = {
    "fp32", "fp16", "bf16", "fp8", "fp4", "fp8_e4m3", "fp8_e5m2", "e8m0", "e4m3", "e2m1",
    "int8", "int16", "int32", "int64", "uint8", "bool", "float", "double", "half",
}


# 任何方法体里都会出现的词，不能拿来证明「这一行指对了」
TRIVIAL_WORDS = {
    "self", "x", "y", "z", "a", "b", "c", "k", "q", "v", "w", "p", "e", "i", "j",
    "n", "m", "t", "s", "h", "d", "r", "B", "T", "S", "N", "D", "H", "L", "O",
    "and", "or", "per", "the", "to", "of", "in", "on", "if", "else", "not",
    "None", "True", "False", "int", "float", "bool", "str", "list", "tuple",
    "len", "range", "print", "return", "for", "while", "with", "as", "is",
    "nn", "torch", "F", "Tensor", "shape", "dim", "dtype", "device",
}


def load_views(page: Path) -> list[dict]:
    text = page.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"window\.DOJO_FLOW_DATA = (\{.*?\});</script>", text, re.S)
    if not m:
        raise SystemExit(f"error: {page} 里没有 DOJO_FLOW_DATA 数据块")
    return json.loads(m.group(1))["views"]


def idents(text: str) -> list[str]:
    """取出文本里的标识符（字母数字下划线，且不以数字开头）。"""
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)


def check_names(views: list[dict], sources: dict[str, str]) -> list[str]:
    """节点名里出现的标识符必须在某份源码里找得到。

    这是防「凭空捏造」的核心检查。实测踩过的坑：节点写了 einsum，而源码里一次 einsum 都没有。
    """
    errors: list[str] = []
    for view in views:
        # 拆节点时，算子第一行写外层调用、内层实参是上一步那个张量节点的 id
        # （如 `self.kv_norm(wkva)` 里的 wkva）。这些 id 是合法名字，源码里没有。
        local_ids = {n["id"] for n in view["nodes"]}
        for node in view["nodes"]:
            for tok in idents(node["name"]):
                if tok in ALLOWED_TOKENS or tok in local_ids:
                    continue
                if any(tok in body for body in sources.values()):
                    continue
                # 允许「检查点键名」这类外部命名：detail 里注明来源即可
                detail = node.get("detail", "")
                if tok in detail and ("检查点" in detail or "键" in detail):
                    continue
                errors.append(
                    f"[{view['id']}/{node['id']}] 节点名 {node['name']!r} 里的 "
                    f"{tok!r} 在源码中找不到")
    return errors


# 公式里 \mathrm{...} 的通用函数/记号：torch、numpy、Python 的常见函数与类型，
# 数学排版里约定俗成的正体词，以及代码里出现、但排版时会改大小写的常量名。
# 不属于这三类的（源码里自定义的函数/类/模块名）必须在某份源码里出现过。
MATH_WORDS = {
    # torch / numpy / Python 的常见函数、类型、方法
    "abs", "absmax", "all", "any", "arange", "argmax", "argmin", "bincount", "cat", "ceil",
    "chunk", "clamp", "clamp_min", "concat", "copysign", "cos", "cumsum", "dot", "einsum",
    "exp", "expand", "flatten", "flip", "float", "floor", "gather", "gelu", "int", "len",
    "linear", "log", "masked_fill", "matmul", "max", "mean", "min", "minlength", "norm",
    "ones", "outer", "pad", "permute", "pow", "prod", "rand", "relu", "repeat", "reshape",
    "roll", "rsqrt", "scatter", "sigmoid", "sign", "silu", "sin", "softmax", "softplus",
    "sort", "split", "sqrt", "squeeze", "stack", "std", "sum", "tanh", "to", "tolist",
    "topk", "transpose", "tril", "triu", "unflatten", "unfold", "unsqueeze", "var", "vec",
    "view", "view_as_complex", "view_as_real", "where", "zeros",
    # 数学排版约定俗成的正体算子（源码里通常没有同名函数）
    "diag", "det", "inv", "rank", "tr", "kron", "cov", "corr", "elementwise", "l1norm",
    "l2norm", "layer_norm", "rms_norm", "layernorm", "rmsnorm", "groupnorm", "batchnorm",
    "logsumexp", "median", "cumprod", "cummax", "hardswish", "mish", "erf", "erfc",
    "sinkhorn",
}
# 比对时统一小写：页面上写 \mathrm{SiLU}、\mathrm{LayerNorm} 这类首字母大写的
# 常见激活/归一化名，不该被判「凭空写的函数名」。
MATH_WORDS_LOWER = {w.lower() for w in MATH_WORDS}


def check_math_names(views: list[dict], sources: dict[str, str], shapes: dict) -> list[str]:
    """折叠与面板的公式里 `\mathrm{X}` 的函数名要有出处。

    通用的 torch/Python 函数名、数学排版约定词不查；其余（源码里自定义的函数、
    类、模块名）必须在某份源码里出现——挡「凭空造个函数名」。
    """
    errors: list[str] = []
    blob = ("\n".join(sources.values()) + "\n" + "\n".join(shapes)).lower()
    for view in views:
        texts = []
        for n in view["nodes"]:
            texts += [s.get("text") or "" for s in n.get("fold") or []]
        for row in view.get("info") or []:
            texts += [v for _k, v in row["rows"]]
        for text in texts:
            for m in re.finditer(r"\\mathrm\{([^}]*)\}", text):
                name = m.group(1).replace("\\_", "_").strip()
                if not name or name.lower() in MATH_WORDS_LOWER:
                    continue
                if name.lower() in blob:
                    continue
                errors.append(f"[{view['id']}] 公式里的「{name}」在源码与形状表里都找不到"
                              "（凭空写的函数名？）")
    return errors


def check_shapes(views: list[dict], shapes: dict) -> tuple[list[str], int]:
    """页面上写的权重形状必须与给定的形状表一致。

    返回 (错误, 比对上的键数)。页面写了「检查点键 X」而 X 不在形状表里，由 resolve()
    报错覆盖，不再另设一份「没比对上的键」清单。
    """
    errors: list[str] = []
    matched = 0
    # 两种写法都认：
    #   ① `embed.weight [129280, 5120]`——键名紧跟形状
    #   ② `$W_e$ [129280, 5120]（检查点键 embed.weight）`——式子用短符号，
    #      检查点键在括号里。这种才是正文里的常见写法。
    # 层号、专家号在两边都写成 `{i}`、`{e}` 占位（材料里也是），比对前统一换成 0。
    NAME_THEN_SHAPE = re.compile(r"([a-z_](?:[a-z0-9_.]|\{[a-z]+\})*)\s*\[([0-9][0-9,\s]*)\]")
    SHAPE_THEN_SRC = re.compile(r"\[([0-9][0-9,\s]*)\]\s*[（(](?:源码里叫|检查点键)\s*([A-Za-z0-9_.{}\-]+)")
    SRC_AFTER_SYMBOL = re.compile(r"\$[^$]+\$\s*\[([0-9][0-9,\s]*)\]\s*[（(](?:源码里叫|检查点键)\s*([A-Za-z0-9_.{}\-]+)")

    def placeholder(txt: str) -> str:
        return re.sub(r"\{[a-z]+\}", "0", txt)

    by_norm = {placeholder(k): k for k in shapes}
    for view in views:
        for node in view["nodes"]:
            # 逐字段核查：键名与它的形状永远写在同一个字段（第三行、或折叠的某一栏）里，
            # 跨字段相邻没有意义。早先把这些字段拼成一个字符串再正则，会把「符号」栏
            # 末尾的 `…源码里叫 norm.weight` 与「说明」栏开头的 `[64]（检查点键 …）`
            # 粘成假的 `norm.weight [64]（检查点键 …）`，凭空报错；为挡它加的启发式
            # 又误伤了 form① 的错键名和真实的带点尾巴。按字段分治从根上消除这类假匹配。
            fields = [node.get("name"), node.get("detail"), node.get("shape"),
                      *(s.get("text") for s in node.get("fold") or [])]

            def compare(key: str, claimed_text: str) -> None:
                nonlocal matched
                claimed = [int(x) for x in claimed_text.replace(" ", "").split(",")]
                expected = shapes[key]
                matched += 1
                if claimed != expected:
                    errors.append(f"[{view['id']}/{node['id']}] {key} 页面写 {claimed}，"
                                  f"材料里是 {expected}")

            seen_here: set[str] = set()
            reported_bad: set[str] = set()

            def resolve(norm_key: str) -> str | None:
                """归一化后的键 → 材料里的原始键。

                页面写了「（检查点键 X）」却对不上材料里的任何键，就是这个形状没被核过，
                按错报——拼错一个字母不能与「形状比对通过」同样是退出码 0。
                """
                # `fp32 [B×T, 384]` 这种方括号左边是 dtype 记号，不是权重键
                if norm_key in DTYPE_WORDS:
                    return None
                if norm_key in by_norm:
                    return by_norm[norm_key]
                # ① 与 ② 两个循环会各解析一次同一个键，同一个错键只报一条
                if norm_key not in reported_bad:
                    reported_bad.add(norm_key)
                    errors.append(f"[{view['id']}/{node['id']}] 检查点键「{norm_key}」不在形状表里"
                                  "（拼错或材料里没有这个键，这个形状没被核过）")
                return None

            for text in fields:
                if not text:
                    continue
                # ① 键名紧跟形状（form①：`gate.weight [16, 32]`）
                for m in NAME_THEN_SHAPE.finditer(text):
                    key = resolve(placeholder(m.group(1)))
                    if key is None:
                        continue
                    compare(key, m.group(2))
                    seen_here.add(key)
                # ② 形状 + （检查点键 X）/（源码里叫 X）
                for rx in (SRC_AFTER_SYMBOL, SHAPE_THEN_SRC):
                    for m in rx.finditer(text):
                        key = resolve(placeholder(m.group(2)))
                        if key is None or key in seen_here:
                            continue
                        compare(key, m.group(1))
                        seen_here.add(key)
                # ③ `$W$ [512, 5120]`：有形状、后面又没跟键注，等于没被核过
                for m in re.finditer(r"\$[^$]+\$\s*\[([0-9][0-9,\s]*)\]", text):
                    after = text[m.end():m.end() + 24]
                    if not re.match(r"\s*[（(](?:检查点键|源码里叫)", after):
                        errors.append(f"[{view['id']}/{node['id']}] 权重形状 [{m.group(1).strip()}]"
                                      " 后面没写「（检查点键 …）」，这个形状没被核过")
    return errors, matched


def check_structure(views: list[dict]) -> list[str]:
    """视图数据自洽：节点 id 不重复，边、分组、批注引用的节点和下钻目标都存在。"""
    errors: list[str] = []
    view_ids = {v["id"] for v in views}
    for v in views:
        ids = [n["id"] for n in v["nodes"]]
        known = set(ids)
        for nid in {i for i in ids if ids.count(i) > 1}:
            errors.append(f"[{v['id']}/{nid}] 节点 id 重复")
        for n in v["nodes"]:
            if n.get("drill") and n["drill"] not in view_ids:
                errors.append(f"[{v['id']}/{n['id']}] 下钻目标 {n['drill']} 不存在")
            shape = (n.get("shape") or "").strip()
            if n.get("kind") == "op" and ("→" in shape or "->" in shape):
                left = re.split(r"→|->", shape, maxsplit=1)[0].strip()
                if not left:
                    errors.append(f"[{v['id']}/{n['id']}] 形状变化缺左边：{shape}")
            for sec in n.get("fold") or []:
                if n.get("kind") != "op":
                    errors.append(f"[{v['id']}/{n['id']}] 折叠只写在算子上")
                    break
                if sec.get("title") not in ("公式", "符号", "说明"):
                    errors.append(f"[{v['id']}/{n['id']}] 折叠的名字只能是「公式」「符号」或「说明」，"
                                  f"现在是 {sec.get('title')!r}")
        for e in v["edges"]:
            for end in (e["from"], e["to"]):
                if end not in known:
                    errors.append(f"[{v['id']}] 边 {e['from']} → {e['to']} 的端点 {end} 不存在")
    return errors


# ---------- 计划文件的源码位置回查 ----------

def plan_rows(text: str):
    """逐张图产出 (标题, 行号, {列名: 值} 或 None)，只读「节点」表。

    ``` 代码围栏里的内容跳过，与 build_page.parse 同一口径——围栏里常贴示例
    计划片段，里面的 `## X.forward` 和节点行不算真图。
    """
    head = sub = None
    cols: list[str] = []
    in_fence = False
    for no, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if s.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if s.startswith("## "):
            head, sub, cols = s[3:].strip(), None, []
            yield head, no, None
        elif s.startswith("### "):
            sub, cols = s[4:].strip(), []
        elif head not in SKIP_HEADS and sub == "节点" and s.startswith("|"):
            cells = split_cells(s)   # 认 \| 转义、$...$ 里的竖线不拆，与生成器同一套
            if not cols:
                cols = cells
            elif not set(s) <= set("|-: "):
                yield head, no, dict(zip(cols, cells))


# 行区间分隔符：半角连字符、en-dash、em-dash、波浪，含全角（手写常敲错）
RANGE = r"[-\u2013\u2014~\uff5e\uff0d]"


def parse_refs(cell: str) -> list[tuple[str, int, int]]:
    """「a.py:1-3，5；b.py:7」→ [(a.py, 1, 3), (a.py, 5, 5), (b.py, 7, 7)]。"""
    out: list[tuple[str, int, int]] = []
    for seg in re.split(r"[；;]", cell):
        fname = ""
        for piece in re.split(r"[，,、]", seg):
            piece = piece.strip()
            m = re.fullmatch(r"(?:(\S+?)[:：])?(\d+)(?:" + RANGE + r"(\d+))?", piece)
            if not m:
                continue
            fname = m.group(1) or fname
            a = int(m.group(2))
            out.append((fname, a, int(m.group(3) or a)))
    return out


def check_plan_sources(plan: Path, root: Path) -> tuple[list[str], int, int]:
    """计划文件的源码位置回查：标题从 def forward 起，节点指的行不是空行或注释，且含节点名。"""
    files: dict[str, list[str] | None] = {}

    def lines_of(name: str) -> list[str] | None:
        if name not in files:
            hits = [root / name] if "/" in name else list(root.rglob(name))
            hits = [h for h in hits if h.is_file()]
            files[name] = hits[0].read_text(encoding="utf-8").splitlines() if len(hits) == 1 else None
        return files[name]

    errors: list[str] = []
    checked = 0
    unchecked = 0
    for head, no, row in plan_rows(plan.read_text(encoding="utf-8")):
        if head in SKIP_HEADS:
            continue
        where = f"第 {no} 行（{head.split('（')[0]}"
        # 这一图标题里写的源码文件。节点指到别的文件多半是张冠李戴：
        # vision 图的节点指到 model.py 里恰好也有同名调用的行，照样能"命中"。
        m_head = re.search(r"[（(](\S+?)[:：]\d+" + RANGE + r"\d+[）)]", head)
        head_file = m_head.group(1) if m_head else None
        if row is None:
            m = re.search(r"[（(](\S+?)[:：](\d+)" + RANGE + r"\d+[）)]", head)
            src = m and lines_of(m.group(1))
            if m and src is not None and "def forward" not in src[int(m.group(2)) - 1]:
                errors.append(f"{where}）标题的起始行 {m.group(1)}:{m.group(2)} 不是 def forward")
            continue
        if "源码" not in row:
            # 节点表整列没写「源码」：row.get 全得空串，整张图的位置检查会空转成
            # 「0 个节点核了名字」还不报错。缺列和缺某一格是两回事，缺列要停下。
            errors.append(f"{where}）节点表没有「源码」列，这一图的位置回查对 0 个节点生效")
            continue
        src_cell = row.get("源码", "")
        refs = parse_refs(src_cell)
        if not refs:
            # 有源码列却解析不出 file:line（「见 m.py:2」、写错格式），这一行就免检了。
            # 解析不出要给信号，不能和「没写」一样静默跳过。
            errors.append(f"{where} / {row.get('id', '?')}）源码列「{src_cell.strip()}」"
                          "解析不出「文件:行」，这一行没核位置")
            continue
        # checked 计的是「有源码列的行」，其中第一行只有短名的那些核不了
        # （下面计进 unchecked）。报出时两者分开，别让 checked 看着像全核过了。
        checked += 1
        if head_file:
            for fname, _a, _b in refs:
                if fname and fname != head_file:
                    errors.append(f"{where} / {row.get('id', '?')}）这一图的源码是 {head_file}，"
                                  f"节点却指到 {fname}")
        where += f" / {row.get('id', '?')}）"
        text = []
        for fname, a, b in refs:
            if not fname:
                # 「源码」列只写行号（与图标题同一文件时可省文件名）→ 用标题里的文件补。
                fname = head_file
                if not fname:
                    errors.append(f"{where}「源码」列只写了行号，但这一图标题里没有"
                                  "「<文件>:<行>」，补上文件名")
                    break
            src = lines_of(fname)
            if src is None:
                errors.append(f"{where}找不到源码文件 {fname}（或同名文件不止一个，"
                              "写上相对 --source-root 的路径）")
                break
            if a > len(src) or not src[a - 1].strip() or src[a - 1].strip().startswith("#"):
                errors.append(f"{where}{fname}:{a} 是空行、注释或超出文件")
            text += [l.split("#")[0] for l in src[a - 1:b]]
        else:
            first = row.get("第一行", "").replace("`", "")
            # 挑出这一行里「有信息量」的标识符：调用名、属性名、类型名。
            # self / x / B / T 这类通用词在任何方法体里都有，不能拿来证明行指对了。
            names = {n for n in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", first)
                     if n not in TRIVIAL_WORDS}
            # 张量第一行常只是一个短名（h、x、q、k），全是通用词，位置核不了。
            # 这类节点交给审查者人眼核（试过拿第三行当锚点，但那是说明文字，误报多）。
            if not names:
                unchecked += 1
                continue
            blob = "\n".join(text)
            # 整名优先：`self.q_norm` 里的 q_norm、`Embedding` 这样的类型名，直接找
            def hit(n: str) -> bool:
                if n in blob:
                    return True
                # 合并层节点写成 `32 × vision.Block`：图上加模块前缀避开重名，
                # 源码里只有 Block。去掉前缀再找。
                tail = n.split(".")[-1]
                if tail != n and tail in blob:
                    return True
                return any(alias in blob for alias in NAME_ALIASES.get(n, ()))
            matched = any(hit(n) for n in names)
            # 「40 × Block」这类合并层节点：类名本身要出现（源码里是 `for layer in self.layers`
            # 那种写法时，类名只在构造处出现，所以也认 `layer(`）
            m = re.fullmatch(r"\d+\s*×\s*([A-Za-z_][A-Za-z0-9_.]*)", first.strip())
            if not matched and m:
                matched = m.group(1).split(".")[-1] in blob
            if not matched:
                errors.append(f"{where}源码 {row.get('源码')} 里找不到第一行的名字"
                              f"（{ '、'.join(sorted(names)) }）")
    return errors, checked, unchecked


def _refs_ok() -> bool:
    return (parse_refs("m.py:1、2、3") == [("m.py", 1, 1), ("m.py", 2, 2), ("m.py", 3, 3)]
            and parse_refs("m.py:1-3，5；b.py:7") == [("m.py", 1, 3), ("m.py", 5, 5), ("b.py", 7, 7)])


def selftest() -> list[tuple[str, bool]]:
    """用虚构的页面数据核对结构、名字、形状三条规则。"""
    checks: list[tuple[str, bool]] = []
    views = [{
        "id": "main", "label": "Toy", "title": "Toy",
        "nodes": [
            {"id": "ids", "name": "input_ids", "kind": "tensor", "shape": "int64 [B, T]"},
            {"id": "emb", "name": "embed_tokens", "kind": "op", "shape": "16 → 16",
             "fold": [{"title": "公式", "text": "$h = w[i]$"}]},
            {"id": "w", "name": "w", "kind": "op", "shape": "16 → 16"},
        ],
        "edges": [{"from": "ids", "to": "emb"}],
    }]
    checks.append(("自洽的视图没有结构错误", not check_structure(views)))

    bad = json.loads(json.dumps(views))
    bad[0]["edges"].append({"from": "ids", "to": "nope"})
    checks.append(("边端点不存在会报出", any("的端点 nope 不存在" in e for e in check_structure(bad))))

    bad = json.loads(json.dumps(views))
    bad[0]["nodes"][0]["fold"] = [{"title": "公式", "text": "x"}]
    checks.append(("折叠写在张量上会报出", any("折叠只写在算子上" in e for e in check_structure(bad))))

    bad = json.loads(json.dumps(views))
    bad[0]["nodes"][1]["shape"] = "→ 16"
    checks.append(("形状箭头缺左边会报出", any("形状变化缺左边" in e for e in check_structure(bad))))

    src = {"m.py": "def forward(self, x):\n    return self.embed_tokens(self.input_ids)\n"}
    checks.append(("名字在源码里找到就不报", not check_names(views[:1], src)))
    checks.append(("名字不在源码里会报出",
                   any("embed_tokens" in e for e in check_names(views, {"m.py": "def forward(): pass"}))))

    shape_views = [{
        "id": "main", "label": "Toy", "title": "Toy",
        "nodes": [{"id": "w", "name": "w", "kind": "op", "shape": "16 → 16",
                   "detail": "gate.weight [16, 32]"}],
        "edges": [],
    }]
    errs, matched = check_shapes(shape_views, {"gate.weight": [16, 32]})
    checks.append(("形状对得上不报", not errs and matched == 1))
    errs2, matched2 = check_shapes(shape_views, {"gate.weight": [8, 8]})
    checks.append(("形状不一致会报出", matched2 == 1 and bool(errs2)))
    e3, matched3 = check_shapes(shape_views, {"other.weight": [1]})
    checks.append(("写的键不在形状表里要报错（这个形状没被核过）", matched3 == 0 and bool(e3)))

    # 正文里的实际写法：`$W$ [512, 5120]（检查点键 layers.{i}.attn.wkv.weight）`
    sym_views = [{
        "id": "main", "label": "Toy", "title": "Toy",
        "nodes": [{"id": "w", "name": "w", "kind": "op", "shape": "16 → 16",
                   "detail": "$W$ [512, 5120]（检查点键 layers.{i}.attn.wkv.weight）"}],
        "edges": [],
    }]
    shapes = {"layers.{i}.attn.wkv.weight": [512, 5120]}
    e4, m4 = check_shapes(sym_views, shapes)
    checks.append(("短符号 + 检查点键能对上", not e4 and m4 == 1))
    e5, m5 = check_shapes(
        [{"id": "main", "label": "T", "title": "T", "edges": [],
          "nodes": [{"id": "w", "name": "w", "kind": "op", "shape": "16 → 16",
                     "detail": "$W$ [512, 9999]（检查点键 layers.{i}.attn.wkv.weight）"}]}],
        shapes)
    checks.append(("短符号写法形状不一致会报出", m5 == 1 and bool(e5)))
    errs6, m6 = check_shapes(
        [{"id": "main", "label": "T", "title": "T", "edges": [],
          "nodes": [{"id": "b", "name": "b", "kind": "op", "shape": "fp32 [B, T]",
                     "detail": "fp32"}]}],
        shapes)
    checks.append(("dtype 记号不算形状键", not errs6))
    e7, m7 = check_shapes(
        [{"id": "main", "label": "T", "title": "T", "edges": [],
          "nodes": [{"id": "w", "name": "w", "kind": "op", "shape": "16 → 16",
                     "detail": "$W$ [512, 5120]（检查点键 no_such.key）"}]}],
        shapes)
    checks.append(("检查点键不在形状表里要报错", bool(e7)))
    e8, m8 = check_shapes(
        [{"id": "main", "label": "T", "title": "T", "edges": [],
          "nodes": [{"id": "w", "name": "w", "kind": "op", "shape": "16 → 16",
                     "detail": "$W$ [512, 5120]"}]}],
        shapes)
    checks.append(("权重形状没写检查点键要报错", bool(e8)))
    # 同一个错键在「键名紧跟形状」和「短符号+键名」两条路径各解析一次，只该报一条
    e_dup, _ = check_shapes(
        [{"id": "main", "label": "T", "title": "T", "edges": [],
          "nodes": [{"id": "w", "name": "w", "kind": "op", "shape": "16 → 16",
                     "detail": "$W$ [16, 16]（检查点键 gate.wieght）"}]}],
        shapes)
    # 形态③：裸形状 + 键注（无 $符号$）。检查③的护栏只针对 $...$，这条走 SHAPE_THEN_SRC
    e_f3, m_f3 = check_shapes(
        [{"id": "main", "label": "T", "title": "T", "edges": [],
          "nodes": [{"id": "n", "name": "n", "kind": "op", "shape": "16 → 16",
                     "detail": "可学习参数 [64]（检查点键 layers.{i}.attn.attn_sink）"}]}],
        {"layers.{i}.attn.attn_sink": [64]})
    checks.append(("裸形状 + 键注能对上", not e_f3 and m_f3 == 1))
    # 同一形态，形状前以标识符结尾（符号栏格式的尾巴）不该把那个标识符当键名
    e_f3b, m_f3b = check_shapes(
        [{"id": "main", "label": "T", "title": "T", "edges": [],
          "nodes": [{"id": "n", "name": "attn_sink", "kind": "op", "shape": "16 → 16",
                     "detail": "[64]（检查点键 layers.{i}.attn.attn_sink）"}]}],
        {"layers.{i}.attn.attn_sink": [64]})
    checks.append(("裸形状 + 键注：前面是标识符也不误报键名", not e_f3b and m_f3b == 1))
    # 反向：form① 键名拼错、形状后面又跟了有效键注，拼错的键名仍要报
    e_f3c, m_f3c = check_shapes(
        [{"id": "main", "label": "T", "title": "T", "edges": [],
          "nodes": [{"id": "n", "name": "n", "kind": "op", "shape": "16 → 16",
                     "detail": "gate.wieght [16, 32]（检查点键 gate.weight）"}]}],
        {"gate.weight": [16, 32]})
    checks.append(("带点的错键名后面跟键注也要报", bool(e_f3c)))
    # 跨字段假匹配：符号栏以「源码里叫 norm.weight」结尾、说明栏以裸形状+键注开头，
    # 两者拼接后曾粘成假的 `norm.weight [64]（检查点键 …）`，凭空报错。逐字段核查后不再发生。
    e_x, m_x = check_shapes(
        [{"id": "main", "label": "T", "title": "T", "edges": [],
          "nodes": [{"id": "n", "name": "core", "kind": "op", "shape": "16 → 16",
                     "fold": [{"title": "符号", "text": "$w$：权重，源码里叫 norm.weight"},
                              {"title": "说明",
                               "text": "[64]（检查点键 layers.{i}.attn.attn_sink）"}]}]}],
        {"layers.{i}.attn.attn_sink": [64], "norm.weight": [5120]})
    checks.append(("跨字段不粘连：符号栏尾巴与说明栏形状不互认", not e_x and m_x == 1))
    checks.append(("同一个错键只报一条",
                   len([e for e in e_dup if "gate.wieght" in e]) == 1))

    # 公式里的函数名要有出处
    mviews = [{"id": "main", "label": "Toy", "title": "Toy", "edges": [],
               "nodes": [{"id": "n", "name": "n", "kind": "op", "shape": "16 → 16",
                          "fold": [{"title": "公式", "text": "$y = \mathrm{softmax}(x)$"}]}]}]
    checks.append(("公式里的通用函数名不报",
                   not check_math_names(mviews, {"m.py": "def forward(): pass"}, {})))
    # 矩阵对角化这类数学排版正体词源码里通常没有同名函数，不能误伤
    diag_m = [{"id": "main", "label": "Toy", "title": "Toy", "edges": [],
               "nodes": [{"id": "n", "name": "n", "kind": "op", "shape": "16 → 16",
                          "fold": [{"title": "公式", "text": "$y = \mathrm{diag}(x)\\odot x$"}]}]}]
    checks.append(("公式里的数学排版词（diag 等）不报",
                   not check_math_names(diag_m, {"m.py": "def forward(): pass"}, {})))
    # 首字母大写的常见激活/归一化名不该被判「凭空写」
    caps_m = [{"id": "main", "label": "Toy", "title": "Toy", "edges": [],
               "nodes": [{"id": "n", "name": "n", "kind": "op", "shape": "16 → 16",
                          "fold": [{"title": "公式", "text": "$y = \mathrm{SiLU}(x)$"}]}]}]
    checks.append(("公式里的首字母大写算子名（SiLU）不报",
                   not check_math_names(caps_m, {"m.py": "def forward(): pass"}, {})))
    bad_m = [{"id": "main", "label": "Toy", "title": "Toy", "edges": [],
              "nodes": [{"id": "n", "name": "n", "kind": "op", "shape": "16 → 16",
                         "fold": [{"title": "公式", "text": "$y = \mathrm{zzq_fake}(x)$"}]}]}]
    checks.append(("公式里凭空写的函数名要报",
                   bool(check_math_names(bad_m, {"m.py": "def forward(): pass"}, {}))))
    checks.append(("公式里的函数名在源码里有就不报",
                   not check_math_names(bad_m, {"m.py": "def forward():\n    zzq_fake(x)"}, {})))


    # check_plan_sources：源码位置回查的判据——名字要真的出现在所指的行里
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "m.py").write_text(
            "def forward(self, x):\n"
            "    qr = self.q_norm(self.wq_a(x))\n"
            "    q = self.wq_b(qr).unflatten(-1, (4, 8))\n", encoding="utf-8")
        plan = root / "dataflow.md"

        def run(plan_text: str):
            plan.write_text(plan_text, encoding="utf-8")
            return check_plan_sources(plan, root)[0]

        good = ("## M.forward（m.py:1-3）\n\n### 节点\n\n"
                "| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |\n|---|---|---|---|---|---|\n"
                "| qr | 算子 | self.q_norm(qra) | 16 → 16 | | m.py:2 |\n")
        checks.append(("计划源码位置：指对了不报", not run(good)))
        checks.append(("计划源码位置：指到相邻的错误行会报出",
                       bool(run(good.replace("m.py:2", "m.py:3")))))
        checks.append(("计划源码位置：捏造的名字会报出",
                       bool(run(good.replace("self.q_norm(qra)", "self.zzz_fake(x)")))))
        checks.append(("计划源码位置：只有 self 这类通用词不算命中",
                       bool(run(good.replace("self.q_norm(qra)", "self.nope(x)")))))
        checks.append(("计划源码位置：方法名换成别处存在的名也要报",
                       bool(run(good.replace("self.q_norm(qra)", "self.wq_b(qra)")))))
        # 合并层节点 `2 × Block`：类名要在所指行里
        merged = ("## M.forward（m.py:1-3）\n\n### 节点\n\n"
                  "| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |\n|---|---|---|---|---|---|\n"
                  "| bl | 算子 | 2 × Block | 16 → 16 | | m.py:1-3 |\n")
        checks.append(("计划源码位置：合并层节点类名对得上",
                       not run(merged.replace("2 × Block", "2 × forward"))))
        checks.append(("计划源码位置：合并层节点伪造类名要报",
                       bool(run(merged.replace("2 × Block", "2 × TotallyFake")))))
        # 节点指到另一个文件（vision 图的节点指 model.py）
        cross = ("## M.forward（m.py:1-3）\n\n### 节点\n\n"
                 "| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |\n|---|---|---|---|---|---|\n"
                 "| qr | 算子 | self.q_norm(qra) | 16 → 16 | | m.py:2 |\n")
        (root / "other.py").write_text("def forward(self, x):\n    return self.q_norm(x)\n",
                                       encoding="utf-8")
        checks.append(("计划源码位置：节点指到另一份源码要报",
                       bool(run(cross.replace("m.py:2", "other.py:2")))))
        # 含转义竖线 \| 的节点行：拆列要认它（与生成器同一套），否则整行位置免检
        esc = ("## M.forward（m.py:1-3）\n\n### 节点\n\n"
               "| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |\n|---|---|---|---|---|---|\n"
               "| qr | 算子 | a \\| b \\| self.q_norm(x) | 16 → 16 | | m.py:2 |\n")
        checks.append(("计划源码位置：含转义竖线的行也核位置",
                       not run(esc) and bool(run(esc.replace("m.py:2", "m.py:9999")))))
        # 顿号分隔的行号
        checks.append(("计划源码位置：顿号分隔的行号能解析",
                       _refs_ok()))
        # 行区间分隔符：半角、en-dash、em-dash、波浪都认
        checks.append(("计划源码位置：六种行区间分隔符都能解析",
                       all(parse_refs(f"m.py:1{c}3") == [("m.py", 1, 3)]
                           for c in "-\u2013\u2014~\uff5e\uff0d")))
        # 全角冒号是手写常见写法，该认；不能因此免检，也不能误报
        checks.append(("计划源码位置：全角冒号的源码列能解析",
                       not run(good.replace("m.py:2", "m.py：2"))))
        # 源码列有内容却解析不出文件:行（前面带字、写错格式）：报错，不能静默免检
        checks.append(("计划源码位置：源码列解析不出要报错",
                       bool(run(good.replace("| m.py:2 |", "| 见 m.py:2 |")))))
        # 节点表整列没有「源码」：位置回查空转，必须报错
        checks.append(("计划源码位置：节点表缺源码列要报错",
                       bool(run(good.replace("| 第三行 | 源码 |", "| 第三行 |")
                                    .replace("|---|---|---|---|---|---|",
                                             "|---|---|---|---|---|")
                                    .replace("| 16 → 16 | | m.py:2 |", "| 16 → 16 | |")))))

        # 第一行只有短名：报「核不了」的计数
        short = ("## M.forward（m.py:1-3）\n\n### 节点\n\n"
                 "| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |\n|---|---|---|---|---|---|\n"
                 "| h | 张量 | h | bf16 [B] | | m.py:2 |\n")
        with tempfile.TemporaryDirectory() as tmp2:
            f2 = Path(tmp2) / "d.md"
            f2.write_text(short, encoding="utf-8")
            _, _, unc = check_plan_sources(f2, root)
        checks.append(("计划源码位置：短名节点计入核不了", unc == 1))
        checks.append(("计划源码位置：只写行号时按图标题的文件补全", not run(good.replace("m.py:2", "2"))))
        # 标题行区间写成 en-dash、冒号写成全角：手写常见，不该据此判「标题没文件名」
        checks.append(("计划源码位置：标题用 en-dash 也能补全文件名",
                       not run(good.replace("m.py:2", "2").replace("m.py:1-3", "m.py:1\u20133"))))
        checks.append(("计划源码位置：标题用全角冒号也能补全文件名",
                       not run(good.replace("m.py:2", "2").replace("m.py:1-3", "m.py\uff1a1-3"))))
        checks.append(("计划源码位置：只写行号补全后指错行仍报出",
                       bool(run(good.replace("m.py:2", "3")))))
        # ``` 围栏里贴的示例图不算真图（与 build_page 同一口径）
        fenced = ("## M.forward（m.py:1-3）\n\n### 节点\n\n"
                  "| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |\n|---|---|---|---|---|---|\n"
                  "| qr | 算子 | self.q_norm(qra) | 16 → 16 | | m.py:2 |\n\n"
                  "```markdown\n## Fake.forward（m.py:1-3）\n\n### 节点\n\n"
                  "| id | 类型 | 第一行 | 第二行 | 第三行 | 源码 |\n|---|---|---|---|---|---|\n"
                  "| z | 算子 | self.zzz_fake(x) | 1 → 2 | | m.py:9 |\n```\n")
        checks.append(("计划源码位置：代码围栏里的示例图不算真图", not run(fenced)))

    return checks


def main() -> int:
    ap = argparse.ArgumentParser(description="数据流页事实核查")
    ap.add_argument("page", type=Path, help="wiki/<name>/index.html 或 research/dataflow.md")
    ap.add_argument("--source", action="append", default=[], metavar="[LABEL=]PATH",
                    help="权威源码，可多次传入。带 LABEL 时只有视图标题里含该 LABEL 才用它校验")
    ap.add_argument("--shapes", type=Path, help="形状表 JSON：{tensor_key: [dim...]}")
    ap.add_argument("--names-only", action="store_true", help="只做节点名核查")
    ap.add_argument("--source-root", type=Path, metavar="DIR",
                    help="传入计划文件时，按「源码」列到这个目录下找文件回查")
    args = ap.parse_args()

    if not args.page.exists():
        print(f"error: 页面不存在: {args.page}", file=sys.stderr)
        return 2

    if args.page.suffix == ".md":
        if args.source or args.shapes or args.names_only:
            print("error: 传计划文件（.md）时只用 --source-root；"
                  "--source/--shapes/--names-only 是对页面（.html）的核查，换个命令跑",
                  file=sys.stderr)
            return 2
        if not args.source_root or not args.source_root.is_dir():
            print("error: 检查计划文件要用 --source-root 指定源码目录", file=sys.stderr)
            return 2
        errors, checked, unchecked = check_plan_sources(args.page, args.source_root)
        if errors:
            print(f"源码位置核查失败：{len(errors)} 处")
            for e in errors:
                print(f"  - {e}")
            return 1
        note = (f"；另有 {unchecked} 个节点第一行只有短名（如 h、x），位置核不了、靠人看"
                if unchecked else "")
        print(f"源码位置核查通过：{checked - unchecked} 个节点核了名字{note}")
        return 0

    if args.shapes and not args.shapes.exists():
        print(f"error: 形状表不存在: {args.shapes}", file=sys.stderr)
        return 2

    views = load_views(args.page)

    if args.names_only and args.shapes:
        print("error: --names-only 与 --shapes 互斥（只核名字就不会核形状），"
              "去掉其中一个再跑", file=sys.stderr)
        return 2
    if args.source_root:
        print("error: --source-root 只用于计划文件（.md），核查页面用 --source "
              "指名源码文件", file=sys.stderr)
        return 2
    if not args.source and not args.shapes:
        print("error: 什么都没核：页面核查要至少给一个 --source 或 --shapes",
              file=sys.stderr)
        return 2

    sources: dict[str, str] = {}
    labeled: list[tuple[str, str]] = []
    labeled_files: set[str] = set()
    for raw in args.source:
        label, sep, path = str(raw).partition("=")
        if not sep:
            label, path = "", label
        else:
            # 标签首尾空白一律去掉；去掉后为空（`=path`、`  =path`）就当没带标签。
            # 否则 ' block =L2.py' 的 ' block' 既不匹配视图、又会静默退回，成漏检。
            label = label.strip()
        p = Path(path)
        if not p.exists():
            print(f"error: 源码不存在: {p}", file=sys.stderr)
            return 2
        if not p.is_file():
            print(f"error: --source 要指向源码文件，不是目录: {p}", file=sys.stderr)
            return 2
        body = p.read_text(encoding="utf-8", errors="ignore")
        sources[path] = body
        if label:
            labeled.append((label, body))
            labeled_files.add(path)

    if args.names_only and not sources:
        print("error: --names-only 仍需 --source 才能比对", file=sys.stderr)
        return 2

    labeled_bodies = {label: body for label, body in labeled}
    universal = [body for path, body in sources.items() if path not in labeled_files]

    errors: list[str] = check_structure(views)
    notes: list[str] = []
    for view in views:
        for node in view["nodes"]:
            if node.get("kind") != "op":
                continue
            if not any(s.get("title") == "公式" for s in node.get("fold") or []):
                notes.append(f"[{view['id']}/{node['id']}] 算子还没有公式")
    if sources:
        for view in views:
            heading = (view.get("title", "") + " " + view.get("label", "")).lower()
            picked = [body for label, body in labeled_bodies.items() if label.lower() in heading]
            if not picked:
                # 没有标签命中这个视图：用不带标签的源码比对。若所有源码都带了标签
                # （universal 为空），退回用全部源码，免得这一视图一份都不核。
                picked = universal or list(sources.values())
            if picked:
                errors += check_names([view], {"view": "\n".join(picked)})
    shape_matched = 0
    shapes: dict = {}
    if not args.names_only and args.shapes:
        # 形状表读不出来就报错退出，不要带着半截数据继续核查
        try:
            shapes = json.loads(args.shapes.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            print(f"error: 形状表无法解析: {args.shapes}（{error}）", file=sys.stderr)
            return 2
        shape_errors, shape_matched = check_shapes(views, shapes)
        errors += shape_errors
        if shape_matched == 0:
            errors.append("形状表一个键都没比对上：页面里要写完整键名，如 model.layers.0.mlp.gate_proj.weight [..]")

    # 公式里的函数名要有出处（通用的 torch/Python 名不查）
    if sources and not args.names_only:
        errors += check_math_names(views, sources, shapes)

    for note in notes:
        print(f"  提示：{note}")

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
        checked.append(f"形状 vs 材料 {shape_matched} 个键")
    print(f"核查通过：{len(views)} 视图 / {nodes} 节点" + (
        "（" + "、".join(checked) + "）" if checked else ""))
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(report(selftest(), "核查脚本"))
    sys.exit(main())
