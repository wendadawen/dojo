"""生成页面主图所需的 SVG 坐标与片段，全部由真实 config 驱动，避免手工估算坐标。"""
from __future__ import annotations
import json

CFG = json.load(open("/tmp/glm53f/config.json"))
T = CFG["text_config"]
LT, MLT = T["layer_types"], T["mlp_layer_types"]
L = T["num_hidden_layers"]

# ---------- 图A：45 层类型条带 ----------
# viewBox 0 0 680 H；左侧留 62 给标签，右侧留 8
X0, X1 = 62, 672
W = (X1 - X0) / L          # 每层宽度
print(f"# 图A: 每层宽度 {W:.4f}")

rows = []
for i in range(L):
    x = X0 + i * W
    kda = LT[i] == "linear_attention"
    rows.append((i, round(x, 2), round(W, 2), kda, MLT[i] == "sparse"))

def band(y, h, pick, cls_true, cls_false):
    out = []
    run_start, run_val = 0, pick(0)
    for i in range(1, L + 1):
        v = pick(i) if i < L else None
        if v != run_val:
            x = X0 + run_start * W
            w = (i - run_start) * W
            cls = cls_true if run_val else cls_false
            out.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{h}" class="{cls}"/>')
            run_start, run_val = i, v
    return out

print("\n# 注意力类型条带（KDA=真）")
for r in band(46, 26, lambda i: LT[i] == "linear_attention", "s-kda", "s-dsa"):
    print(r)
print("\n# FFN 类型条带（sparse=真）")
for r in band(84, 26, lambda i: MLT[i] == "sparse", "s-moe", "s-dense"):
    print(r)

print("\n# 层号刻度（每 4 层一个，落在 DSA 层上）")
ticks = [i for i in range(L) if LT[i] == "deepseek_sparse_attention"]
for i in ticks:
    cx = X0 + (i + 0.5) * W
    print(f'<text x="{cx:.1f}" y="130" class="t-tick">{i}</text>')
print(f'<text x="{X0 + 0.5*W:.1f}" y="130" class="t-tick">0</text>')
print(f'<text x="{X0 + (L-1+0.5)*W:.1f}" y="130" class="t-tick">44</text>')

# ---------- 校验 ----------
n_kda = sum(1 for t in LT if t == "linear_attention")
n_dsa = L - n_kda
print(f"\n# KDA {n_kda} DSA {n_dsa}；DSA 层号 {ticks}")
print(f"# dense 层 {[i for i in range(L) if MLT[i]=='dense']}")
assert ticks == [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]
