# Hy4-Preview 前向数据流逐部件数值对撞（mini model = 官方 modeling exec，架构零改动）。
# 方法：forward hook 捕获官方实现的输入输出 -> 用本脚本独立复算同一公式 -> 逐张量对撞。
# 输出存档 probe_forward.out，页面数字一律取自本存档。
#
# 覆盖：
#   [A] iHC：attn_hc / ffn_hc / hc_head 的 pre/post 门控与混合输出手工复算对撞；
#       初始 4 条流 == embeds 的验证；门控数值范围。
#   [B] DSA indexer：layer1（full）分数手工复算（自写 RoPE）+ top-k 集合对撞；
#       shared 层（2,3,4 / 6,7,8）top-k 与前一 full 层完全一致的验证。
#   [C] MLA 注意力：q/kv 投影、潜向量展开、因果+稀疏掩码、sink 拼接 softmax、
#       sigmoid gate、o_proj 全链路手工复算对撞；sink 吸收的概率质量测量。
#   [D] MoE：路由（sigmoid+bias+top-8+归一化+2.827）手工复算对撞（bias 置非零）；
#       专家 SwiGLU（clamp 10）逐 token 复算对撞 + clamp 单元验证；
#       共享专家直通；dense 层 0 MLP 对撞。
#   [E] KV cache 口径：mini 实测形状 + 真实 config 算术（HF eager 展开 K/V vs vLLM 潜向量 MQA）。
#   [F] MTP：按 vLLM 语义（enorm/hnorm/eh_proj -> 无 iHC 单流块 -> final_layernorm ->
#       复用 lm_head）搭 mini MTP 前向并跑通，验证 draft 块自带 full indexer。
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_model import (small_config, HYV4ForCausalLM, HYV4Attention, HYV4MoE,
                        HYV4MLP, HYV4RMSNorm, HYV4RotaryEmbedding)

torch.manual_seed(0)
cfg = small_config()
model = HYV4ForCausalLM(cfg)
model.eval()
H = cfg.hidden_size
MC = cfg.hc_mult

T = 24
seq = torch.randint(0, 1000, (1, T))

# ---------------- 捕获容器 ----------------
cap = {}          # name -> list of (inputs, outputs)


def pre_hook(name):
    def h(m, args, kwargs):
        cap.setdefault(name, []).append({"args": args, "kwargs": kwargs})
    return h


def post_hook(name):
    def h(m, args, output):
        cap.setdefault(name + ":out", []).append(output)
    return h


L1 = model.model.layers[1]
L1.attn_hc.register_forward_pre_hook(pre_hook("L1.attn_hc"), with_kwargs=True)
L1.attn_hc.register_forward_hook(post_hook("L1.attn_hc"))
L1.ffn_hc.register_forward_pre_hook(pre_hook("L1.ffn_hc"), with_kwargs=True)
L1.ffn_hc.register_forward_hook(post_hook("L1.ffn_hc"))
model.model.hc_head.register_forward_pre_hook(pre_hook("hc_head"), with_kwargs=True)
model.model.hc_head.register_forward_hook(post_hook("hc_head"))
L1.self_attn.register_forward_pre_hook(pre_hook("L1.self_attn"), with_kwargs=True)
L1.self_attn.register_forward_hook(post_hook("L1.self_attn"))
L1.mlp.register_forward_pre_hook(pre_hook("L1.mlp"), with_kwargs=True)
L1.mlp.register_forward_hook(post_hook("L1.mlp"))
model.model.layers[0].mlp.register_forward_pre_hook(pre_hook("L0.mlp"), with_kwargs=True)
model.model.layers[0].mlp.register_forward_hook(post_hook("L0.mlp"))

# 路由 bias 置非零（mini 默认零初始化，无法行使 bias 路径）
with torch.no_grad():
    for li in range(cfg.num_hidden_layers):
        mlp = model.model.layers[li].mlp
        if isinstance(mlp, HYV4MoE):
            mlp.gate.e_score_correction_bias.copy_(
                torch.linspace(-0.04, 0.05, cfg.n_routed_experts))

# 每层 top-k 输出捕获（DecoderLayer 返回 (hidden, topk_indices)）
layer_topk = {}


def topk_hook(li):
    def h(m, args, output):
        layer_topk[li] = output[1].detach().clone()
    return h


for li, layer in enumerate(model.model.layers):
    layer.register_forward_hook(topk_hook(li))

with torch.no_grad():
    out = model(seq, use_cache=False)
print(f"[setup] mini forward ok: logits {tuple(out.logits.shape)}, "
      f"nan={torch.isnan(out.logits).any().item()}")


def diff(name, a, b):
    d = (a.float() - b.float()).abs().max().item()
    print(f"    collision {name}: max_abs_diff = {d:.3e}")
    return d


# ============ [A] iHC 门控对撞 ============
print("\n[A] iHC hyper-connections (official HYV4HyperConnection vs manual recompute)")


def manual_hc(mod, X):
    """复刻 HYV4HyperConnection.forward（modeling L647-666）。"""
    flat = X.flatten(2).float()                                  # [B,S,mc*H]
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + mod.input_norm.eps)
    mixes = F.linear(flat, mod.fn.float()) * rsqrt               # [B,S,2*mc]
    pre_b, post_b = mod.base.split(mod.hc_mult, dim=-1)
    pre_scale, post_scale = mod.scale.unbind(0)
    pre_logits, post_logits = mixes.split(mod.hc_mult, dim=-1)
    pre = torch.sigmoid(pre_logits * pre_scale + pre_b) + mod.hc_eps
    post = mod.hc_post_magnitude * torch.sigmoid(post_logits * post_scale + post_b) + mod.hc_eps
    y = torch.sum(pre.unsqueeze(-1) * X, dim=2)
    return pre, post, y


for name in ["L1.attn_hc", "L1.ffn_hc"]:
    X = cap[name][0]["args"][0]                                  # [B,S,mc,H]
    post_off, y_off = cap[name + ":out"][0]
    pre_m, post_m, y_m = manual_hc(getattr(L1, name.split(".")[1]), X)
    diff(f"{name} mixed output (pre-gated sum)", y_m, y_off)
    diff(f"{name} post gates", post_m, post_off)
    if name == "L1.attn_hc":
        print(f"    pre gates: min={pre_m.min():.4f} max={pre_m.max():.4f} mean={pre_m.mean():.4f}")
        print(f"    post gates: min={post_m.min():.4f} max={post_m.max():.4f} mean={post_m.mean():.4f}")

Xh = cap["hc_head"][0]["args"][0]
head_out_off = cap["hc_head:out"][0]
flat = Xh.flatten(2).float()
rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + model.model.hc_head.input_norm.eps)
mixes = F.linear(flat, model.model.hc_head.hc_fn.float()) * rsqrt
pre = torch.sigmoid(mixes * model.model.hc_head.hc_scale + model.model.hc_head.hc_base) + model.model.hc_head.eps
head_out_m = torch.sum(pre.unsqueeze(-1) * Xh, dim=2)
diff("hc_head merged output", head_out_m, head_out_off)
print(f"    hc_head gates: min={pre.min():.4f} max={pre.max():.4f} mean={pre.mean():.4f}")

# 初始 4 条流 == embeds
embeds = model.model.embed_tokens(seq)
stream_in = cap["L1.attn_hc"][0]["args"][0]  # layer1 输入 = layer0 输出流，非初始流；改从 layer0 前捕获
# 用 layer0 的 attn_hc 输入验证：layer0 的输入流就是初始流
cap0 = {}
model.model.layers[0].attn_hc.register_forward_pre_hook(
    lambda m, a, k: cap0.update(X=a[0]), with_kwargs=True)
with torch.no_grad():
    model(seq, use_cache=False)
X0 = cap0["X"]
same = all((X0[:, :, i] - embeds).abs().max().item() == 0.0 for i in range(MC))
print(f"    initial {MC} streams identical to embeddings at layer0 input: {same}")

# ============ [B] DSA indexer 对撞 ============
print("\n[B] DSA lightning indexer (layer 1, full) — manual recompute with hand-written RoPE")
idx = L1.self_attn.indexer
kw = cap["L1.self_attn"][0]["kwargs"]
x_attn = kw["hidden_states"]                     # DecoderLayer 全 kwargs 调用 attention；输入 = input_layernorm 输出
pos_emb = kw["position_embeddings"]              # (cos, sin) [B,S,rope]
attn_mask_full = kw["attention_mask"]            # [B,1,S,T]
topk_off = cap["L1.self_attn:out"][0][2]         # 官方 topk indices [B,S,topk]

# 手工 indexer：q = wq_b(q_resid)，q_resid 需要从 attention 内部重算（q_a_proj(q_a_ln)）
with torch.no_grad():
    q_resid = L1.self_attn.q_a_layernorm(L1.self_attn.q_a_proj(x_attn))
    B, S, _ = x_attn.shape
    q = idx.wq_b(q_resid).view(B, S, idx.n_heads, idx.head_dim)
    q_pass, q_rot = torch.split(q, [idx.head_dim - idx.qk_rope_head_dim, idx.qk_rope_head_dim], dim=-1)
    k_raw = idx.wk(x_attn)
    k_ln = idx.k_norm(k_raw.float()).to(x_attn.dtype)          # LayerNorm（fp32 计算）
    k_pass, k_rot = torch.split(k_ln.unsqueeze(2), [idx.head_dim - idx.qk_rope_head_dim, idx.qk_rope_head_dim], dim=-1)
    cos, sin = pos_emb
    # 自写 RoPE（与官方 apply_rotary_pos_emb 相同公式，独立实现）
    def my_rope(x, cos, sin):
        c = cos.unsqueeze(2)                                    # [B,S,1,R]
        s = sin.unsqueeze(2)
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return x * c + torch.cat((-x2, x1), dim=-1) * s
    q_rot = my_rope(q_rot, cos, sin)
    k_rot = my_rope(k_rot, cos, sin)                # [B,S,1,R]，与 cos.unsqueeze(2) 逐元素（官方 unsqueeze_dim=2 语义）
    q_full = torch.cat([q_pass, q_rot], dim=-1)                # [B,S,IH,ID]
    k_full = torch.cat([k_pass, k_rot], dim=-1).squeeze(2)     # [B,S,ID]
    scores = torch.relu(torch.einsum("bshd,btd->bsht", q_full.float(), k_full.float()))
    w = idx.weights_proj(x_attn).float() * (idx.n_heads ** -0.5) * idx.softmax_scale
    index_scores_m = torch.einsum("bsh,bsht->bst", w, scores)
    index_scores_m = index_scores_m + attn_mask_full[:, 0]     # 加性因果掩码
    topk_m = index_scores_m.topk(min(idx.index_topk, S), dim=-1).indices

sets_eq = all(set(topk_m[0, t].tolist()) == set(topk_off[0, t].tolist()) for t in range(S))
print(f"    manual top-k sets == official: {sets_eq} (topk={topk_off.shape[-1]}, "
      f"index_n_heads={idx.n_heads}, index_head_dim={idx.head_dim}, "
      f"softmax_scale={idx.softmax_scale:.6f}={idx.head_dim}**-0.5)")
nz = index_scores_m[index_scores_m > 0]
print(f"    manual index scores: >0 fraction={(nz.numel() / index_scores_m.numel()):.3f}, "
      f"max={index_scores_m.max().item():.4f}")
relu0 = (scores == 0).float().mean().item()
print(f"    per-head ReLU exact-zero fraction (head scores): {relu0:.3f}")

# shared 层复用
print("    shared-layer top-k reuse (official DecoderLayer outputs):")
reuse_ok = True
for grp_start in (1, 5, 9):
    for li in range(grp_start + 1, min(grp_start + 4, cfg.num_hidden_layers)):
        same_l = torch.equal(layer_topk[li], layer_topk[grp_start])
        reuse_ok &= same_l
print(f"    layers 2,3,4 reuse layer1; 6,7,8 reuse layer5; (mini 10-layer): {reuse_ok}")
print(f"    layer0 own indexer: {model.model.layers[0].self_attn.indexer is not None}; "
      f"layer2 indexer is None: {model.model.layers[2].self_attn.indexer is None}")

# ============ [C] MLA 注意力全链路对撞 ============
print("\n[C] MLA attention with sink + sigmoid gate (manual recompute vs official)")
attn = L1.self_attn
with torch.no_grad():
    # gate
    gate_states = attn.gate_proj(x_attn).view(B, S, -1, attn.gate_projection_size)
    # q
    q2 = attn.q_b_proj(q_resid).view(B, S, attn.num_heads, attn.qk_head_dim).transpose(1, 2)
    q_pass2, q_rot2 = torch.split(q2, [attn.qk_nope_head_dim, attn.qk_rope_head_dim], dim=-1)
    # kv 潜向量
    compressed = attn.kv_a_proj_with_mqa(x_attn)
    kv_pass, k_rot2 = torch.split(compressed, [attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1)
    k_pass2 = attn.kv_a_layernorm(kv_pass).view(B, 1, S, attn.kv_lora_rank)
    k_rot2 = k_rot2.view(B, 1, S, attn.qk_rope_head_dim)
    c2, s2 = pos_emb
    c2, s2 = c2.unsqueeze(1), s2.unsqueeze(1)                  # [B,1,S,R]
    q_rot2 = q_rot2 * c2 + torch.cat((-q_rot2[..., 8:], q_rot2[..., :8]), dim=-1) * s2
    k_rot2 = k_rot2 * c2 + torch.cat((-k_rot2[..., 8:], k_rot2[..., :8]), dim=-1) * s2
    q_states = torch.cat((q_pass2, q_rot2), dim=-1)            # [B,H,S,Dk]
    # 展开
    kv_nope = attn.kv_b_proj(k_pass2).view(B, S, -1, attn.qk_nope_head_dim + attn.v_head_dim).transpose(1, 2)
    k_nope, v_states = torch.split(kv_nope, [attn.qk_nope_head_dim, attn.v_head_dim], dim=-1)
    k_rot_e = k_rot2.expand(-1, k_nope.shape[1], -1, -1)
    key_states = torch.cat([k_nope, k_rot_e], dim=-1)          # [B,H,S,Dk]
    # logits + 掩码（因果 + top-k 稀疏）
    logits = torch.matmul(q_states, key_states.transpose(2, 3)) * attn.scaling
    mask = attn_mask_full.clone()
    index_mask = (
        topk_off.new_ones((B, S, key_states.shape[2]), dtype=torch.bool)
        .scatter(-1, topk_off.long(), False).unsqueeze(1)
    )
    mask = mask.masked_fill(index_mask, torch.finfo(x_attn.dtype).min)
    logits = logits + mask
    # sink 拼接 softmax
    sinks = attn.sinks.reshape(1, -1, 1, 1).expand(B, -1, S, -1)
    combined = torch.cat([logits, sinks], dim=-1)
    combined = combined - combined.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined, dim=-1, dtype=combined.dtype)
    scores_p = probs[..., :-1]
    sink_mass = probs[..., -1]                                 # sink 吸收的质量
    attn_out = torch.matmul(scores_p, v_states).transpose(1, 2)
    attn_out = attn_out * torch.sigmoid(gate_states)
    attn_out = attn_out.reshape(B, S, -1)
    y_m = attn.o_proj(attn_out)

y_off = cap["L1.self_attn:out"][0][0]
diff("attention block output (after o_proj)", y_m, y_off)
attn_w_off = cap["L1.self_attn:out"][0][1]
diff("attention probs (sink-dropped, eager return)", scores_p, attn_w_off)
print(f"    sink absorbed prob mass: min={sink_mass.min():.3e} mean={sink_mass.mean():.3e} "
      f"max={sink_mass.max():.3e} (per head/query)")
print(f"    sink parameter values: min={attn.sinks.min():.4f} max={attn.sinks.max():.4f} "
      f"(init={cfg.learnable_sink_init})")
print(f"    sigmoid gate values: min={torch.sigmoid(gate_states).min():.4f} "
      f"max={torch.sigmoid(gate_states).max():.4f} mean={torch.sigmoid(gate_states).mean():.4f}")
print(f"    scaling = qk_head_dim**-0.5 = {attn.qk_head_dim}**-0.5 = {attn.scaling:.6f}")

# ============ [D] MoE 路由 + 专家对撞 ============
print("\n[D] MoE routing + experts (manual recompute vs official)")
moe = L1.mlp
x_moe = cap["L1.mlp"][0]["args"][0]                            # post_attention_layernorm 输出
y_moe_off = cap["L1.mlp:out"][0]
with torch.no_grad():
    # 路由（fp32）
    xr = x_moe.view(-1, H)
    router_logits = F.linear(xr.float(), moe.gate.weight.float())
    r_scores = router_logits.sigmoid()
    r_bias = moe.gate.e_score_correction_bias
    scores_choice = r_scores + r_bias
    # n_group=1 / topk_group=1：组机制退化为全专家
    assert cfg.n_group == 1 and cfg.topk_group == 1
    topk_i = scores_choice.topk(cfg.num_experts_per_tok, dim=-1)[1]
    topk_w = r_scores.gather(1, topk_i)
    if cfg.norm_topk_prob:
        topk_w = topk_w / (topk_w.sum(-1, keepdim=True) + 1e-20)
    topk_w = topk_w * cfg.routed_scaling_factor
    # 专家 + 共享
    y = torch.zeros_like(xr)
    for t in range(xr.shape[0]):
        for e, w in zip(topk_i[t].tolist(), topk_w[t].tolist()):
            gu = F.linear(xr[t], moe.experts.gate_up_proj[e])
            g, u = gu.chunk(2, dim=-1)
            g = g.clamp(max=cfg.swiglu_limit)
            u = u.clamp(min=-cfg.swiglu_limit, max=cfg.swiglu_limit)
            y[t] += F.silu(g) * u * w @ moe.experts.down_proj[e].T
    shared = moe.shared_experts(x_moe)
    y_m = (y.view(*x_moe.shape) + shared)
diff("MoE output (routed + shared)", y_m, y_moe_off)
t0 = 5
print(f"    token {t0}: top-8 experts = {sorted(topk_i[t0].tolist())}")
print(f"    token {t0}: normalized weights (pre-scaling) sum = "
      f"{(topk_w[t0] / cfg.routed_scaling_factor).sum().item():.6f}")
print(f"    token {t0}: weights after *routed_scaling_factor({cfg.routed_scaling_factor}) = "
      f"[{', '.join(f'{v:.4f}' for v in topk_w[t0].tolist())}]")
print(f"    e_score_correction_bias: min={r_bias.min():.4f} max={r_bias.max():.4f} "
      f"(nonzero in this test: {r_bias.abs().max().item() > 0})")

# clamp 单元验证
with torch.no_grad():
    gu = torch.tensor([[20.0, -20.0, 7.0, -7.0, 0.5, -0.5]])
    g, u = gu.chunk(2, dim=-1)
    g = g.clamp(max=cfg.swiglu_limit)
    u = u.clamp(min=-cfg.swiglu_limit, max=cfg.swiglu_limit)
    act = F.silu(g) * u
print(f"    swiglu clamp unit: input gate=[20,-20,7] up=[-7,0.5,-0.5] -> "
      f"gate_clamped={g.tolist()} up_clamped={u.tolist()} out={[f'{v:.4f}' for v in act[0].tolist()]}")

# dense 层 0
x0_mlp = cap["L0.mlp"][0]["args"][0]
y0_off = cap["L0.mlp:out"][0]
with torch.no_grad():
    y0_m = model.model.layers[0].mlp.down_proj(
        F.silu(model.model.layers[0].mlp.gate_proj(x0_mlp)) * model.model.layers[0].mlp.up_proj(x0_mlp))
diff("dense layer0 MLP output", y0_m, y0_off)
print(f"    layer0 mlp type: {type(model.model.layers[0].mlp).__name__} "
      f"(intermediate_size={cfg.intermediate_size}); layer1: {type(moe).__name__}")

# ============ [E] KV cache 口径 ============
print("\n[E] KV cache: mini measured shapes + real-config arithmetic")
with torch.no_grad():
    o2 = model(seq, use_cache=True)
pkv = o2.past_key_values
k1, v1 = pkv.key_cache[1], pkv.value_cache[1]
print(f"    mini HF-eager cache: K={tuple(k1.shape)} V={tuple(v1.shape)} "
      f"(B,H,T,D; D_k=qk_nope+rope={cfg.qk_nope_head_dim}+{cfg.qk_rope_head_dim}, D_v=v_head={cfg.v_head_dim})")
print(f"    mini indexer cache (full layers only): {tuple(pkv.indexer_cache[1].shape)} "
      f"(B,T,index_head_dim={cfg.index_head_dim}); cached layers = {sorted(pkv.indexer_cache)}")

real = json.load(open("config.json"))
rH = real["num_attention_heads"]
r_nope, r_rope, r_v = real["qk_nope_head_dim"], real["qk_rope_head_dim"], real["v_head_dim"]
r_kv, r_L = real["kv_lora_rank"], real["num_hidden_layers"]
r_ixd, r_topk = real["index_head_dim"], real["index_topk"]
hf_elems = rH * (r_nope + r_rope) + rH * r_v
vllm_elems = r_kv + r_rope
print(f"    real config: HF eager per-token-per-layer = {rH}*({r_nope}+{r_rope}) + {rH}*{r_v} "
      f"= {hf_elems} elems = {hf_elems*2} B (bf16)")
print(f"    real config: vLLM latent MQA per-token-per-layer = {r_kv}+{r_rope} = {vllm_elems} elems "
      f"= {vllm_elems*2} B (bf16) + per-head sinks {rH} params (once, fp32)")
print(f"    main-cache expansion ratio = {hf_elems}/{vllm_elems} = {hf_elems/vllm_elems:.2f}x")
print(f"    per token over {r_L} layers (main cache, bf16): HF {hf_elems*2*r_L/1024/1024:.2f} MB "
      f"vs vLLM {vllm_elems*2*r_L/1024:.1f} KB")
full_layers = sum(1 for t in real["indexer_types"] if t == "full")
print(f"    indexer cache: {full_layers} full layers x {r_ixd} dims/token "
      f"(vLLM fp8: ~{full_layers*(r_ixd + r_ixd//128*4)} B/token; HF eager bf16: "
      f"{full_layers*r_ixd*2} B/token)")

# ============ [F] MTP mini（vLLM 语义） ============
print("\n[F] MTP draft layer (vLLM semantics: enorm/hnorm/eh_proj -> no-iHC block -> final_ln -> shared lm_head)")


class MiniMTP(nn.Module):
    """按 vllm mtp.py L339-341,359-366,369-389 与 model.py L162-185 拼装。
    使用官方 HYV4Attention / HYV4MoE / HYV4RMSNorm 模块，残差语义照抄 vLLM 单流路径。"""

    def __init__(self, cfg, layer_idx, lm_head):
        super().__init__()
        self.enorm = HYV4RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.hnorm = HYV4RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.eh_proj = nn.Linear(cfg.hidden_size * 2, cfg.hidden_size, bias=False)
        self.self_attn = HYV4Attention(cfg, layer_idx)
        self.input_layernorm = HYV4RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = HYV4RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = HYV4MoE(cfg) if cfg.mlp_layer_types[layer_idx] == "sparse" else HYV4MLP(cfg)
        self.final_layernorm = HYV4RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = lm_head

    def forward(self, embeds, prev_hidden, position_embeddings, attention_mask):
        e = self.enorm(embeds)
        h = self.hnorm(prev_hidden)
        x = self.eh_proj(torch.cat([e, h], dim=-1))          # mtp.py L380-382
        residual = x
        a, _, mtp_topk = self.self_attn(                      # model.py L172-177 单流
            hidden_states=self.input_layernorm(x),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask)
        x = a + residual                                       # model.py L179
        residual = x
        f = self.mlp(self.post_attention_layernorm(x))        # model.py L181-183
        final = self.final_layernorm(f + residual)            # mtp.py L388 (RMSNorm(h+r))
        return self.lm_head(final), mtp_topk, final


cfg_mtp = small_config()
cfg_mtp.indexer_types = cfg_mtp.indexer_types + ["full"]       # MTP 层自带 full indexer
cfg_mtp.mlp_layer_types = cfg_mtp.mlp_layer_types + ["sparse"]
torch.manual_seed(1)
mtp = MiniMTP(cfg_mtp, layer_idx=cfg.num_hidden_layers, lm_head=model.lm_head).eval()

with torch.no_grad():
    trunk_out = model.model(seq)                               # last_hidden_state = norm(hc_head(·))
    prev_hidden = trunk_out.last_hidden_state
    embeds2 = model.model.embed_tokens(seq)
    rotary = HYV4RotaryEmbedding(cfg_mtp)
    pos_ids = torch.arange(T).unsqueeze(0)
    pos_emb2 = rotary(embeds2, position_ids=pos_ids)
    S2 = T
    keep = torch.ones(S2, S2, dtype=torch.bool).tril()
    mask2 = torch.where(keep, torch.zeros(()), torch.full((), torch.finfo(torch.float32).min)).float()[None, None]
    mtp_logits, mtp_topk, mtp_final = mtp(embeds2, prev_hidden, pos_emb2, mask2)

print(f"    trunk last_hidden (MTP input 'previous_hidden_states'): {tuple(prev_hidden.shape)}")
print(f"    eh_proj input: cat([enorm(embeds), hnorm(prev_hidden)]) -> {2*H} -> output {H}")
print(f"    MTP block output hidden: {tuple(mtp_final.shape)}; draft logits: {tuple(mtp_logits.shape)}")
print(f"    MTP block own indexer (full): {mtp.self_attn.indexer is not None}, "
      f"topk shape={tuple(mtp_topk.shape)}")
own_topk = not torch.equal(mtp_topk, layer_topk[1])
print(f"    MTP top-k differs from trunk layer1 top-k (own selection): {own_topk}")
shared_lm = mtp.lm_head.weight.data_ptr() == model.lm_head.weight.data_ptr()
print(f"    draft lm_head IS trunk lm_head (shared weight): {shared_lm}")

print("\n[done] all collisions above are between official exec'd code and independent manual recompute")
