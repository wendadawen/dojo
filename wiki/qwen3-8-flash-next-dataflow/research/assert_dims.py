"""从 config 推导各模块维度，并与真实张量 shape 逐项断言。

任何不一致会直接 AssertionError，因此本脚本通过即说明推导与 checkpoint 完全一致。
"""
import json

H = json.load(open("/tmp/qwen38fn/headers.json"))
C = json.load(open("/tmp/qwen38fn/config.json"))["text_config"]

d = C["hidden_size"]
checks = []


def chk(desc, name, expect):
    got = H[name]["shape"]
    assert got == expect, f"MISMATCH {name}: expect {expect}, got {got}"
    checks.append((desc, name, expect))


P = "model.language_model.layers"

# ---- GDN 线性注意力（第 0 层，linear_attention）----
nk, nv = C["linear_num_key_heads"], C["linear_num_value_heads"]
dk, dv = C["linear_key_head_dim"], C["linear_value_head_dim"]
key_dim, value_dim = nk * dk, nv * dv
conv_dim = 2 * key_dim + value_dim
chk("GDN in_proj_qkv = 2*key_dim + value_dim", f"{P}.0.linear_attn.in_proj_qkv.weight", [conv_dim, d])
chk("GDN in_proj_z = value_dim", f"{P}.0.linear_attn.in_proj_z.weight", [value_dim, d])
chk("GDN in_proj_b = num_v_heads", f"{P}.0.linear_attn.in_proj_b.weight", [nv, d])
chk("GDN in_proj_a = num_v_heads", f"{P}.0.linear_attn.in_proj_a.weight", [nv, d])
chk("GDN conv1d depthwise 在 conv_dim 通道", f"{P}.0.linear_attn.conv1d.weight",
    [conv_dim, 1, C["linear_conv_kernel_dim"]])
chk("GDN A_log 每个 v head 一个", f"{P}.0.linear_attn.A_log", [nv])
chk("GDN dt_bias 每个 v head 一个", f"{P}.0.linear_attn.dt_bias", [nv])
chk("GDN 门控 norm 作用在 head_v_dim", f"{P}.0.linear_attn.norm.weight", [dv])
chk("GDN out_proj: value_dim -> d", f"{P}.0.linear_attn.out_proj.weight", [d, value_dim])

# ---- QSA 全注意力（第 3 层）----
nh, nkv, hd = C["num_attention_heads"], C["num_key_value_heads"], C["head_dim"]
chk("QSA q_proj 输出 2x（query + gate）", f"{P}.3.self_attn.q_proj.weight", [nh * hd * 2, d])
chk("QSA k_proj = num_kv_heads * head_dim", f"{P}.3.self_attn.k_proj.weight", [nkv * hd, d])
chk("QSA v_proj = num_kv_heads * head_dim", f"{P}.3.self_attn.v_proj.weight", [nkv * hd, d])
chk("QSA o_proj: nh*hd -> d", f"{P}.3.self_attn.o_proj.weight", [d, nh * hd])
chk("QSA q_norm 作用在 head_dim", f"{P}.3.self_attn.q_norm.weight", [hd])
ih, inh, ikv = C["indexer_head_dim"], C["indexer_n_heads"], C["indexer_kv_heads"]
chk("indexer qk 合并投影 = (n_heads+kv_heads)*head_dim",
    f"{P}.3.self_attn.indexer.index_qk_proj.weight", [(inh + ikv) * ih, d])
chk("indexer q/k norm 作用在 indexer_head_dim", f"{P}.3.self_attn.indexer.q_layernorm.weight", [ih])

# ---- MoE ----
E, im, si = C["num_experts"], C["moe_intermediate_size"], C["shared_expert_intermediate_size"]
chk("路由专家 gate_up 打包（2*im）", f"{P}.0.mlp.experts.gate_up_proj", [E, 2 * im, d])
chk("路由专家 down", f"{P}.0.mlp.experts.down_proj", [E, d, im])
chk("路由器权重", f"{P}.0.mlp.gate.weight", [E, d])
chk("共享专家 gate_proj", f"{P}.0.mlp.shared_expert.gate_proj.weight", [si, d])
chk("共享专家门（标量）", f"{P}.0.mlp.shared_expert_gate.weight", [1, d])

# ---- Gated Residual（超连接）----
hc, lr = C["hc_count"], C["hc_lowrank"]
chk("超连接 norm 作用在 hc_count*d", f"{P}.0.attn_hyper_connection.hc_norm.weight", [hc * d])
chk("超连接 down: hc*d -> lowrank", f"{P}.0.attn_hyper_connection.input_mix_weight_down.weight", [lr, hc * d])
chk("超连接 up: lowrank -> hc*d", f"{P}.0.attn_hyper_connection.input_mix_weight_up.weight", [hc * d, lr])
chk("注入权重每流一个标量", f"{P}.0.attn_hyper_connection.block_inject_weight.weight", [hc, hc * d])
chk("末端 mixer 无 block_inject（只 3 个张量）",
    "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight", [lr, hc * d])
assert "model.language_model.hyper_connection_mixer.block_inject_weight.weight" not in H, \
    "末端 mixer 不应有 block_inject_weight"
checks.append(("末端 mixer 确实没有 block_inject_weight（use_combine=False）", "(缺失即正确)", None))

# ---- PLE ----
ple_d = C["ple_embed_dim"]
ngram_heads = (C["ngram_size"] - 1) * C["heads_per_ngram"]
head_dim_per_ngram = ple_d // ngram_heads
chk("PLE key_proj: ple_embed_dim -> hc*d", f"{P}.1.ple.key_proj.weight", [hc * d, ple_d])
chk("PLE value_proj: ple_embed_dim -> d", f"{P}.1.ple.value_proj.weight", [d, ple_d])
chk("PLE 空洞深度卷积在 hc*d 通道", f"{P}.1.ple.conv1d.weight", [hc * d, 1, C["ple_conv_kernel_size"]])
chk("每个 n-gram head 的向量维度", f"{P}.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
    [2500012, head_dim_per_ngram])
chk("n-gram head 数", f"{P}.1.ple.ple_embedding.ngram_heads_vocab_sizes", [ngram_heads])

for desc, name, exp in checks:
    print(f"OK  {desc}")
    print(f"      {name} = {exp}")
print()
print(f"全部 {len(checks)} 项 config->张量 shape 断言通过")
print()
print("推导量：")
print(f"  key_dim = {nk}*{dk} = {key_dim}    value_dim = {nv}*{dv} = {value_dim}")
print(f"  GDN conv_dim = 2*{key_dim}+{value_dim} = {conv_dim}")
print(f"  v/k head 比 = {nv}/{nk} = {nv // nk}  (源码用 repeat_interleave 把 q/k 扩到 {nv} 头)")
print(f"  QSA: {nh} q 头 / {nkv} kv 头 -> GQA group = {nh // nkv}, head_dim={hd}")
rot = int(hd * C["rope_parameters"]["partial_rotary_factor"])
print(f"  rotary_dim = {hd} * {C['rope_parameters']['partial_rotary_factor']} = {rot}  (<= indexer_head_dim {ih}: {rot <= ih})")
print(f"  indexer: block_topk = budget/compress = {C['indexer_budget']}/{C['indexer_compress_ratio']} = {C['indexer_budget'] // C['indexer_compress_ratio']}")
print(f"  PLE: n-gram head 数 = ({C['ngram_size']}-1)*{C['heads_per_ngram']} = {ngram_heads}, 每 head 维度 = {ple_d}/{ngram_heads} = {head_dim_per_ngram}")
print(f"  PLE 空洞卷积感受野 = (kernel-1)*dilation = ({C['ple_conv_kernel_size']}-1)*{C['ngram_size']} = {(C['ple_conv_kernel_size'] - 1) * C['ngram_size']}")
