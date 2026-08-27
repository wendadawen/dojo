"""
探针 5：端到端前向 —— 用官方 DecoderLayer 堆叠成 8 层缩小模型，跑 prefill + 逐 token decode，
用 forward hook 抓每一步的真实张量形状，作为数据流图的直接依据。

缩小的只有：层数 45→8、专家数 288→16、词表（不建 embedding，直接喂随机 embedding）。
保留的结构特征：KDA/DSA 按 4 层一循环交替、前 3 层 dense 后续 sparse、
hc_mult=4 四路残差流、indexer full、hidden_size/head_dim/kv_lora_rank 全部真实值。

验证目标
  1. 主干数据流的每一步真实形状（图的依据）
  2. 8 层 45 层同构：层类型序列取自真实 config 前 8 项
  3. prefill 后 decode 的状态延续正确（KDA 循环状态 + DSA kv cache 同时工作）
  4. 全程无 NaN/Inf
  5. 四路残差流在逐层前进中是否真的分化（初始四路相同）

对应源码
  Glm5NextTextDecoderLayer.forward   modeling_glm5_next.py L1280-1329
  Glm5NextTextModel.forward          L1431-1494
"""
from __future__ import annotations
import sys, json, math
sys.path.insert(0, "/tmp/glm53f/probe")
import torch
import torch.nn as nn
from harness import (real_config, shrink_config, banner, Glm5NextTextDecoderLayer,
                     Glm5NextTextHyperHead, Glm5NextTextRMSNorm, MiniCache)

torch.manual_seed(0)
DT = torch.float32
NL = 8
C = shrink_config(num_layers=NL)
CF = real_config()

banner(f"探针 5：端到端前向（{NL} 层缩小模型，结构特征与 45 层一致）")
print(f"      layer_types      = {C.layer_types}")
print(f"      mlp_layer_types  = {C.mlp_layer_types}")
print(f"      indexer_types    = {C.indexer_types}")
print(f"      hidden_size={C.hidden_size} hc_mult={C.hc_mult} "
      f"linear_num_heads={C.linear_num_heads} num_attention_heads={C.num_attention_heads}")
print(f"      真实 45 层的类型序列前 8 项 = {CF.layer_types[:NL]}  与上面一致="
      f"{CF.layer_types[:NL]==C.layer_types}")

layers = nn.ModuleList([Glm5NextTextDecoderLayer(C, i) for i in range(NL)]).to(DT).eval()
norm = Glm5NextTextRMSNorm(C.hidden_size, C.rms_norm_eps).to(DT).eval()
hc_head = Glm5NextTextHyperHead()

# 严格按官方 Glm5NextPreTrainedModel._init_weights（L1362-1401）初始化，
# 不能只初始化 dim>=2 的张量——mHC 的 base/scale 是 1 维，且由 torch.empty 创建，
# 漏掉会读到未初始化内存。
with torch.no_grad():
    for m in layers.modules():
        cls = type(m).__name__
        if cls == "Glm5NextTextHyperConnection":
            m.fn.normal_(0.0, 0.02)      # init.normal_(fn, 0, 0.02)
            m.base.zero_()               # init.zeros_(base)
            m.scale.fill_(1.0)           # init.ones_(scale)
        elif cls == "Glm5NextTextForgetGate":
            m.A_log.zero_()              # safe 分支：init.zeros_(A_log)
            m.dt_bias.uniform_(math.log(1e-3), math.log(1e-1))
            dt = m.dt_bias.exp().clamp_min(1e-4)
            m.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))   # 逆 softplus
        elif cls == "Glm5NextTextExperts":
            m.gate_up_proj.normal_(0.0, C.initializer_range)
            m.down_proj.normal_(0.0, C.initializer_range)
        elif cls == "Glm5NextTextTopkRouter":
            m.weight.normal_(0.0, C.initializer_range)
            m.e_score_correction_bias.zero_()
        elif cls == "Glm5NextTextIndexer":
            m.index_kpool_compress_ape.zero_()
            m.index_kpool_compress_gate.fill_(1.0)
        elif isinstance(m, nn.Linear):
            m.weight.normal_(0.0, C.initializer_range)
            if m.bias is not None:
                m.bias.zero_()
        elif isinstance(m, nn.Conv1d):
            m.weight.normal_(0.0, C.initializer_range)
        elif cls in ("Glm5NextTextRMSNorm", "Glm5NextTextRMSNormGated"):
            m.weight.fill_(1.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.fill_(1.0)
            m.bias.zero_()

nparam = sum(p.numel() for p in layers.parameters())
print(f"      缩小模型参数量 = {nparam:,} = {nparam/1e9:.3f} B（仅供跑通，不代表真实规模）")

# ---------- hook：抓每个子模块的输入输出形状 ----------
trace = []


def mk(name):
    def hook(m, inp, out):
        def shp(t):
            if torch.is_tensor(t):
                return tuple(t.shape)
            if isinstance(t, (tuple, list)):
                return [shp(u) for u in t if torch.is_tensor(u)] or None
            return None
        trace.append((name, shp(inp[0] if inp else None), shp(out)))
    return hook


handles = []
for i, l in enumerate(layers):
    tag = "KDA" if C.layer_types[i] == "linear_attention" else "DSA"
    mlp_tag = "MoE" if C.mlp_layer_types[i] == "sparse" else "denseMLP"
    handles.append(l.attn_hc.register_forward_hook(mk(f"L{i}.attn_hc")))
    handles.append(l.input_layernorm.register_forward_hook(mk(f"L{i}.input_layernorm")))
    handles.append(l.self_attn.register_forward_hook(mk(f"L{i}.self_attn[{tag}]")))
    handles.append(l.ffn_hc.register_forward_hook(mk(f"L{i}.ffn_hc")))
    handles.append(l.post_attention_layernorm.register_forward_hook(mk(f"L{i}.post_attn_ln")))
    handles.append(l.mlp.register_forward_hook(mk(f"L{i}.mlp[{mlp_tag}]")))
    handles.append(l.register_forward_hook(mk(f"L{i}.__layer__")))

# ---------- prefill ----------
banner("5.1 prefill 数据流（B=1, S=12）")
B, S = 1, 12
cache = MiniCache(num_layers=NL)
emb = torch.randn(B, S, C.hidden_size, dtype=DT)
mask = torch.ones(B, S, dtype=torch.bool)

# 复刻 Glm5NextTextModel.forward L1477
h = emb.unsqueeze(2).expand(-1, -1, C.hc_mult, -1).contiguous()
print(f"      embed {tuple(emb.shape)} --expand--> hidden_streams {tuple(h.shape)}")
four_same_in = bool(all((h[:, :, i] == emb).all() for i in range(C.hc_mult)))
print(f"      入口四路是否完全相同 = {four_same_in}")

topk = None
with torch.no_grad():
    for i, l in enumerate(layers):
        h, topk = l(h, attention_mask=mask, position_ids=None, past_key_values=cache,
                    use_cache=True, position_embeddings=None, prev_topk_indices=topk)
    out = norm(hc_head(h))

print(f"      末层 hidden_streams {tuple(h.shape)} --hc_head(mean)--> "
      f"{tuple(hc_head(h).shape)} --RMSNorm--> {tuple(out.shape)}")
print(f"      finite = {bool(torch.isfinite(out).all())}   "
      f"输出 std = {float(out.std()):.6f}")

# 四路分化程度
spread = [(h[:, :, i] - h.mean(2)).abs().mean().item() for i in range(C.hc_mult)]
print(f"      出口四路与其均值的平均绝对偏差 = {[f'{v:.4f}' for v in spread]}")
print(f"      四路是否已分化（非全相同）= {not all(torch.allclose(h[:,:,0], h[:,:,i]) for i in range(1,C.hc_mult))}")

print("\n      逐步形状追踪（第 0 层 KDA + 第 3 层 DSA）：")
for name, si, so in trace:
    if name.startswith("L0.") or name.startswith("L3."):
        print(f"        {name:24s} in={si}  out={so}")

# ---------- 5.2 decode ----------
banner("5.2 逐 token decode（延续 prefill 的 cache）")
trace.clear()
with torch.no_grad():
    for step in range(4):
        hd = torch.randn(B, 1, C.hidden_size, dtype=DT).unsqueeze(2).expand(
            -1, -1, C.hc_mult, -1).contiguous()
        tk = None
        for i, l in enumerate(layers):
            hd, tk = l(hd, attention_mask=torch.ones(B, 1, dtype=torch.bool),
                       position_ids=None, past_key_values=cache, use_cache=True,
                       position_embeddings=None, prev_topk_indices=tk)
        od = norm(hc_head(hd))
        kv = cache.layers[3].keys.shape[-2]
        rs = tuple(cache.layers[0].recurrent_states[0].shape)
        print(f"      step {step+1}: 输出 {tuple(od.shape)}  finite={bool(torch.isfinite(od).all())}  "
              f"DSA kv 长度={kv}  KDA 状态={rs}")

print(f"\n      DSA 层 kv cache 随 token 线性增长：S={S} → {cache.layers[3].keys.shape[-2]}")
print(f"      KDA 层循环状态形状恒定：{tuple(cache.layers[0].recurrent_states[0].shape)}（与 token 数无关）")
print(f"      这正是混合架构的目的：只有 {sum(1 for t in CF.layer_types if t=='deepseek_sparse_attention')}/45 层"
      f" 的 KV 随长度增长，其余 {sum(1 for t in CF.layer_types if t=='linear_attention')} 层是常数状态")

# ---------- 5.3 层内数据流顺序 ----------
banner("5.3 单层内部数据流顺序（源码 L1291-1329 的执行序）")
print("""      residual = h                              # [B,S,4,D]
      post, comb, x = attn_hc(h)                # x:[B,S,D]  post:[B,S,4]  comb:[B,S,4,4]
      x = input_layernorm(x)                    # RMSNorm
      x = self_attn(x)                          # KDA 或 DSA(MLA+indexer)
      h = post ⊗ x + comb^T @ residual          # 回写 4 路
      residual = h
      post, comb, x = ffn_hc(h)
      x = post_attention_layernorm(x)
      x = mlp(x)                                # MoE 或 dense MLP
      h = post ⊗ x + comb^T @ residual""")
print(f"      注意：两个站点各有独立的 mHC 参数（attn_hc / ffn_hc），互不共享")
print(f"      两个 RMSNorm 作用在「已压成单路」的张量上（[B,S,{C.hidden_size}]），不是 4 路")

for hd_ in handles:
    hd_.remove()
