"""
验证目标：用真实 checkpoint 的张量清单交叉验证从 config.json + modeling_glm5_next.py
推导出的结构性结论。
对应源码：Glm5NextTextDecoderLayer.__init__（attn_hc/ffn_hc）、
         Glm5NextTextAttention.__init__（indexer 只在 indexer_types=="full" 时创建）、
         Glm5NextTextLinearAttention.__init__（conv1d / forget_gate / b_proj）
对应 config：layer_types / mlp_layer_types / indexer_types / n_routed_experts
判据：每条推导若成立，则对应张量数量必须精确匹配；不匹配即源码理解有偏差。
"""
import json, re
from collections import Counter, defaultdict

H = json.load(open("/tmp/glm53f/headers.json"))
CFG = json.load(open("/tmp/glm53f/config.json"))
T = CFG["text_config"]

tensors = {}
for shard, hdr in H.items():
    for name, meta in hdr.items():
        if name == "__metadata__":
            continue
        tensors[name] = meta

print(f"[0] 张量总数 = {len(tensors)}")

# ---------- 1. 顶层命名空间 ----------
ns = Counter()
for n in tensors:
    if n.startswith("model.language_model.layers."):
        ns["text.layers"] += 1
    elif n.startswith("model.visual."):
        ns["visual"] += 1
    elif n.startswith("model.language_model."):
        ns["text.other"] += 1
    else:
        ns[n] += 1
print("[1] 命名空间分布:", dict(ns))

# ---------- 2. 层号分布 ----------
layer_re = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
by_layer = defaultdict(set)
for n in tensors:
    m = layer_re.match(n)
    if m:
        by_layer[int(m.group(1))].add(m.group(2))
print(f"[2] 出现的层号: min={min(by_layer)} max={max(by_layer)} count={len(by_layer)}")
print(f"    config.num_hidden_layers = {T['num_hidden_layers']}, "
      f"num_nextn_predict_layers = {T['num_nextn_predict_layers']}")
extra = sorted(k for k in by_layer if k >= T["num_hidden_layers"])
print(f"    超出主干的层号(MTP): {extra}")

# ---------- 3. 层类型交叉验证 ----------
lt = T["layer_types"]
kda_marker, dsa_marker = "self_attn.conv1d.weight", "self_attn.kv_a_proj_with_mqa.weight"
mismatch = []
for i in range(T["num_hidden_layers"]):
    has_kda = kda_marker in by_layer[i]
    has_dsa = dsa_marker in by_layer[i]
    want_kda = lt[i] == "linear_attention"
    if has_kda != want_kda or has_dsa == want_kda:
        mismatch.append((i, lt[i], has_kda, has_dsa))
n_kda = sum(1 for i in range(T["num_hidden_layers"]) if kda_marker in by_layer[i])
n_dsa = sum(1 for i in range(T["num_hidden_layers"]) if dsa_marker in by_layer[i])
print(f"[3] 权重实测 KDA 层={n_kda} DSA 层={n_dsa}; "
      f"config 声明 {Counter(lt)}; 冲突={mismatch if mismatch else '无'}")

# ---------- 4. indexer 分布（cross-layer topk 共享） ----------
idx_marker = "self_attn.indexer.wk.weight"
has_idx = sorted(i for i in range(T["num_hidden_layers"]) if idx_marker in by_layer[i])
it = T["indexer_types"]
want_idx = sorted(i for i in range(T["num_hidden_layers"])
                  if lt[i] == "deepseek_sparse_attention" and it[i] == "full")
print(f"[4] 实测带 indexer 的层={has_idx}")
print(f"    config 推导应带 indexer 的层={want_idx}  一致={has_idx == want_idx}")

# ---------- 5. mHC 双站点 ----------
for tag in ["attn_hc", "ffn_hc"]:
    cnt = sum(1 for i in by_layer if f"{tag}.fn" in by_layer[i])
    shapes = {tuple(tensors[f"model.language_model.layers.{i}.{tag}.fn"]["shape"])
              for i in by_layer if f"{tag}.fn" in by_layer[i]}
    print(f"[5] {tag}.fn 层数={cnt} shape集合={shapes}")
hc = T["hc_mult"]
print(f"    源码 mix=(2+hc_mult)*hc_mult = (2+{hc})*{hc} = {(2 + hc) * hc}; "
      f"fn 期望 shape=[{(2 + hc) * hc}, {hc * T['hidden_size']}]")

# ---------- 6. MoE / dense 分布与专家数 ----------
exp_re = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")
exp_per_layer = defaultdict(set)
for n in tensors:
    m = exp_re.match(n)
    if m:
        exp_per_layer[int(m.group(1))].add(int(m.group(2)))
moe_layers = sorted(exp_per_layer)
print(f"[6] 带 routed experts 的层={len(moe_layers)}  层号={moe_layers[:6]}...{moe_layers[-3:]}")
print(f"    每层专家数集合={set(len(v) for v in exp_per_layer.values())}; "
      f"config.n_routed_experts={T['n_routed_experts']}")
mlt = T["mlp_layer_types"]
want_sparse = sorted(i for i in range(T["num_hidden_layers"]) if mlt[i] == "sparse")
print(f"    config sparse 层数={len(want_sparse)} 主干实测={len([i for i in moe_layers if i < 45])} "
      f"一致={want_sparse == [i for i in moe_layers if i < 45]}")
dense_layers = sorted(i for i in range(T["num_hidden_layers"])
                      if "mlp.gate_proj.weight" in by_layer[i])
print(f"    实测 dense MLP 层={dense_layers}; config first_k_dense_replace={T['first_k_dense_replace']}")

# ---------- 7. 关键张量 shape 对照 config ----------
def show(name):
    if name in tensors:
        print(f"    {name}: shape={tensors[name]['shape']} dtype={tensors[name]['dtype']}")
    else:
        print(f"    {name}: (缺失)")

print("[7] 关键张量 shape:")
for n in ["model.language_model.embed_tokens.weight", "lm_head.weight",
          "model.language_model.layers.0.self_attn.q_proj.weight",
          "model.language_model.layers.0.self_attn.conv1d.weight",
          "model.language_model.layers.0.self_attn.forget_gate.A_log",
          "model.language_model.layers.0.self_attn.forget_gate.dt_bias",
          "model.language_model.layers.0.self_attn.b_proj.weight",
          "model.language_model.layers.0.attn_hc.fn",
          "model.language_model.layers.0.attn_hc.base",
          "model.language_model.layers.0.attn_hc.scale",
          "model.language_model.layers.3.self_attn.q_a_proj.weight",
          "model.language_model.layers.3.self_attn.q_b_proj.weight",
          "model.language_model.layers.3.self_attn.kv_a_proj_with_mqa.weight",
          "model.language_model.layers.3.self_attn.kv_b_proj.weight",
          "model.language_model.layers.3.self_attn.o_proj.weight",
          "model.language_model.layers.3.self_attn.indexer.wq_b.weight",
          "model.language_model.layers.3.self_attn.indexer.wk.weight",
          "model.language_model.layers.3.self_attn.indexer.weights_proj.weight",
          "model.language_model.layers.3.self_attn.indexer.index_kpool_compress_ape",
          "model.language_model.layers.3.self_attn.indexer.index_kpool_compress_gate",
          "model.language_model.layers.3.mlp.gate.weight",
          "model.language_model.layers.3.mlp.gate.e_score_correction_bias",
          "model.language_model.layers.3.mlp.experts.0.gate_proj.weight",
          "model.language_model.layers.3.mlp.experts.0.down_proj.weight",
          "model.language_model.layers.3.mlp.shared_experts.gate_proj.weight",
          "model.visual.patch_embed.proj.weight",
          "model.visual.downsample.weight",
          "model.visual.merger.gate_proj.weight"]:
    show(n)

# ---------- 8. dtype 分布 ----------
dt = Counter(v["dtype"] for v in tensors.values())
print(f"[8] dtype 分布: {dict(dt)}")
scale_cnt = sum(1 for n in tensors if n.endswith("weight_scale_inv"))
print(f"    weight_scale_inv 张量数={scale_cnt}; "
      f"quantization_config.weight_block_size={CFG['quantization_config']['weight_block_size']}")

# ---------- 9. MTP 层结构 ----------
print("[9] MTP 层(45) 独有模块:")
mtp = by_layer.get(45, set())
main = by_layer.get(44, set())
print("    仅 MTP 有:", sorted(x for x in mtp - main))
print("    MTP 缺少(相比 44 层):", sorted(x for x in main - mtp)[:10])

# ---------- 10. 参数量统计 ----------
def numel(meta):
    n = 1
    for d in meta["shape"]:
        n *= d
    return n

tot = act_note = 0
groups = defaultdict(int)
for name, meta in tensors.items():
    if name.endswith("weight_scale_inv"):
        continue          # 量化缩放因子不计入参数量
    k = numel(meta)
    tot += k
    if name.startswith("model.visual."):
        groups["visual"] += k
    elif ".layers.45." in name:
        groups["mtp(layer45)"] += k
    elif "mlp.experts." in name:
        groups["routed_experts"] += k
    elif "mlp.shared_experts." in name:
        groups["shared_expert"] += k
    elif "embed_tokens" in name or name == "lm_head.weight":
        groups["embed+lm_head"] += k
    elif "_hc." in name:
        groups["mHC"] += k
    elif ".indexer." in name:
        groups["dsa_indexer"] += k
    elif ".self_attn." in name:
        groups["attn(kda+mla)"] += k
    else:
        groups["other(norms/dense mlp)"] += k

print(f"[10] 参数量合计 = {tot:,}  ({tot / 1e9:.2f} B)")
for k, v in sorted(groups.items(), key=lambda x: -x[1]):
    print(f"     {k:26s} {v:>15,}  {v / tot * 100:5.2f}%")
print(f"     index.json metadata.total_size = {json.load(open('/tmp/glm53f/model.safetensors.index.json'))['metadata']['total_size']:,} 字节(磁盘)")
