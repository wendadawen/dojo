# 验证目标：用 headers.json 交叉验证源码推导的结构论断，并核对参数量。
# 每项输出对照：源码/config 推导值 vs checkpoint 张量实测值。
import json
import re
from collections import Counter, defaultdict

H = json.load(open(__import__("pathlib").Path(__file__).parent / "headers.json"))

DT = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "I64": 8, "I32": 4, "I8": 1}


def numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def nbytes(meta):
    return numel(meta["shape"]) * DT[meta["dtype"]]


# ---------- 1. 层结构验证 ----------
layer_re = re.compile(r"^model\.layers\.(\d+)\.")
layer_ids = sorted({int(m.group(1)) for n in H if (m := layer_re.match(n))})
print(f"[L1] decoder layers in ckpt: {len(layer_ids)} (config 78), "
      f"min={layer_ids[0]} max={layer_ids[-1]}")

# indexer 层（full 层才有 indexer.*）
full_layers = sorted({int(m.group(1)) for n in H
                      if (m := layer_re.match(n)) and ".self_attn.indexer." in n})
print(f"[L2] layers with indexer (full): {len(full_layers)}")
print(f"     full layer ids: {full_layers}")
cfg = json.load(open(__import__("pathlib").Path(__file__).parent / "config.json"))
cfg_full = [i for i, t in enumerate(cfg["indexer_types"]) if t == "full"]
print(f"     config full ids match: {cfg_full == full_layers}")

# dense vs sparse MLP：sparse 层有 experts.gate_up_proj，dense 层有 mlp.gate_proj
moE_layers = sorted({int(m.group(1)) for n in H
                     if (m := layer_re.match(n)) and ".mlp.experts.gate_up_proj" in n})
dense_layers = sorted({int(m.group(1)) for n in H
                       if (m := layer_re.match(n)) and re.search(r"\.mlp\.(gate_proj|up_proj|down_proj)\.", n)})
print(f"[L3] MoE layers: {len(moE_layers)}, dense layers: {dense_layers} (config mlp_layer_types: dense={cfg['mlp_layer_types'].index('sparse')} 第0层dense)")
cfg_dense = [i for i, t in enumerate(cfg["mlp_layer_types"]) if t == "dense"]
print(f"     config dense ids match: {cfg_dense == dense_layers}")

# hc 结构：每层几个 hc 模块、hc_head
hc_pre = sorted({n for n in H if "hc_pre" in n})
hc_post = sorted({n for n in H if "hc_post" in n})
hc_names = sorted(set(re.sub(r"layers\.\d+", "layers.N", n) for n in H if ".hc" in n))
print(f"[L4] hc tensor names sample: {hc_names}")

# ---------- 2. 关键张量 shape 验证 ----------
def show(name_pat, expect=None):
    pat = re.compile(name_pat)
    hits = {n: (H[n]["dtype"], H[n]["shape"]) for n in H if pat.search(n)}
    for n, (d, s) in list(hits.items())[:4]:
        print(f"     {n}: {d} {s}")
    return hits


print("[S1] layer0 attention tensors:")
for pat in [r"^model\.layers\.0\.self_attn\.(q_a_proj|q_b_proj|q_a_layernorm|kv_a_proj_with_mqa|kv_b_proj|o_proj|gate_proj|learnable_sink_param)\.weight$",
            r"^model\.layers\.0\.self_attn\.indexer\.(wq_b|wk|weights_proj|k_norm)\.weight$"]:
    show(pat)
print("[S2] layer0 hc tensors:")
show(r"^model\.layers\.0\.hc_attn_layer\.")
print("[S3] layer0 mlp (dense) vs layer1 mlp (MoE):")
show(r"^model\.layers\.0\.mlp\.")
show(r"^model\.layers\.1\.mlp\.gate\.weight$")
print(f"     layer1 experts.gate_up_proj: {H.get('model.layers.1.mlp.experts.gate_up_proj.weight', {}).get('shape')}")
print(f"     layer1 experts.down_proj: {H.get('model.layers.1.mlp.experts.down_proj.weight', {}).get('shape')}")
print(f"     layer1 shared_experts gate: {H.get('model.layers.1.mlp.shared_experts.gate_proj.weight', {}).get('shape')}")
print(f"     layer1 router e_score_correction_bias-ish: {[n for n in H if 'layers.1.mlp.gate' in n and 'proj' not in n]}")

# ---------- 3. 参数量核对 ----------
groups = defaultdict(int)
for n, meta in H.items():
    cnt = numel(meta["shape"])
    if n.startswith("model.mtp_layers."):
        groups["mtp"] += cnt
    elif ".mlp.experts." in n:
        groups["moe_experts"] += cnt
    elif ".mlp.shared_experts." in n:
        groups["moe_shared"] += cnt
    elif ".mlp.gate." in n or re.search(r"\.mlp\.e_score", n):
        groups["moe_router"] += cnt
    elif re.search(r"\.mlp\.(gate|up|down)_proj\.weight$", n):
        groups["dense_mlp"] += cnt
    elif ".self_attn.indexer." in n:
        groups["dsa_indexer"] += cnt
    elif ".hc" in n or n.startswith("model.hc_head"):
        groups["ihc"] += cnt
    elif ".self_attn." in n:
        groups["attention"] += cnt
    elif ".layernorm" in n or ".norm" in n:
        groups["norms"] += cnt
    elif n == "model.embed_tokens.weight":
        groups["embed"] += cnt
    elif n == "lm_head.weight":
        groups["lm_head"] += cnt
    else:
        groups["other:" + n] += cnt

total = sum(v for k, v in groups.items() if not k.startswith("other"))
print("\n[P] param counts by group:")
for k in sorted(groups):
    if not k.startswith("other"):
        print(f"     {k:>14}: {groups[k]:,}")
    else:
        print(f"     {k}: {groups[k]:,}")
print(f"     {'TOTAL':>14}: {total:,}  ({total/1e9:.2f}B)")

# 激活参数量（单 token，dense层1 + moe层77 各 top-8 专家 + shared + attn + 索引器等）
attn_keys = ["q_a_proj.weight", "q_a_layernorm.weight", "q_b_proj.weight",
             "kv_a_proj_with_mqa.weight", "kv_a_layernorm.weight", "kv_b_proj.weight",
             "o_proj.weight", "linear_gate.weight", "learnable_sink_param"]
attn_per = sum(numel(H["model.layers.1.self_attn." + k]["shape"]) for k in attn_keys)
act = groups["embed"] + groups["lm_head"] + 78 * 2 * 6144 + 6144  # embed+lm_head+norms
act += attn_per * 78
# dense mlp layer 0
act += groups["dense_mlp"]
# moe: top-8 of 256 routed + shared + router
one_expert_gate_up = numel(H["model.layers.1.mlp.experts.gate_up_proj"]["shape"]) // 256
one_expert_down = numel(H["model.layers.1.mlp.experts.down_proj"]["shape"]) // 256
moe_per_layer = 8 * (one_expert_gate_up + one_expert_down)
moe_per_layer += groups["moe_shared"] // 77
moe_per_layer += groups["moe_router"] // 77
act += moe_per_layer * 77
# indexer: 只在 21 个 full 层激活
idx_keys = ["wq_b.weight", "wk.weight", "weights_proj.weight", "k_norm.weight", "k_norm.bias"]
idx_per = sum(numel(H["model.layers.0.self_attn.indexer." + k]["shape"]) for k in idx_keys)
act += idx_per * 21
# ihc
act += groups["ihc"]
print(f"\n[A] per-layer: attn={attn_per:,} indexer={idx_per:,} moe(top8+shared+router)={moe_per_layer:,}")
print(f"[A] activated (incl embed+lm_head, no MTP): {act:,} ({act/1e9:.2f}B)")
act_no_el = act - groups["embed"] - groups["lm_head"]
print(f"[A] activated (excl embed+lm_head, no MTP): {act_no_el:,} ({act_no_el/1e9:.2f}B)")
print(f"[A] activated (excl embed+lm_head, incl MTP): {(act_no_el+groups['mtp']):,} ({(act_no_el+groups['mtp'])/1e9:.2f}B)")
print(f"[A] activated (incl embed+lm_head, incl MTP): {(act+groups['mtp']):,} ({(act+groups['mtp'])/1e9:.2f}B)")
trunk = total - groups["mtp"]
print(f"[A] trunk total (excl MTP): {trunk:,} ({trunk/1e9:.2f}B)")

# MoE 侧理论值对照
hs, mi = cfg["hidden_size"], cfg["moe_intermediate_size"]
theo_expert = 3 * mi * hs
print(f"[C] one routed expert theoretical: 3*{mi}*{hs} = {theo_expert:,}; "
      f"measured gate_up/256+down/256 = {one_expert_gate_up + one_expert_down:,}")
