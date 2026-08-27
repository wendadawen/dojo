"""
探针 4：MoE 路由（noaux_tc / sigmoid 打分）与 dense-MLP 混排。

验证目标
  1. 路由公式：sigmoid 打分（非 softmax）→ 加 e_score_correction_bias → 分组 topk →
     取原始 score → 归一化 → 乘 routed_scaling_factor
  2. topk_weights 的实际取值范围与和：norm_topk_prob=True 时归一化后和为 1，
     再乘 2.5 → 和为 2.5（这点必须实测，不能想当然认为权重和是 1）
  3. n_group=1 / topk_group=1 时分组机制退化为全局 topk —— 实测验证
  4. e_score_correction_bias 只影响「选谁」，不影响「权重多少」（aux-loss-free 的关键）
  5. 共享专家：intermediate = moe_intermediate_size * n_shared_experts，与 routed 并行相加
  6. SwiGLU 的 clamp（swiglu_limit=10.0）：gate 单侧上限、up 双侧限幅
  7. 每 token 激活参数量实测统计
  8. 前 3 层 dense MLP（intermediate_size=12288）vs 后 42 层 MoE

对应源码
  Glm5NextTextTopkRouter.forward   modeling_glm5_next.py L158-183
  Glm5NextTextExperts.forward      L120-135, _apply_gate L137-142
  Glm5NextTextMoE.forward          L200-207
  Glm5NextTextMLP.forward          L98-104
"""
from __future__ import annotations
import sys, json
sys.path.insert(0, "/tmp/glm53f/probe")
import torch
import torch.nn.functional as F
from harness import (real_config, shrink_config, banner, Glm5NextTextTopkRouter,
                     Glm5NextTextMoE, Glm5NextTextMLP, Glm5NextTextExperts)

torch.manual_seed(0)
C = real_config()
DT = torch.float32
banner("探针 4：MoE 路由（noaux_tc）与 dense/sparse 混排")

# ---------- 4.1 路由器配置 ----------
router = Glm5NextTextTopkRouter(C).to(DT).eval()
print("[4.1] 路由器参数与配置")
print(f"      weight {tuple(router.weight.shape)} = (n_routed_experts, hidden_size)")
print(f"      e_score_correction_bias {tuple(router.e_score_correction_bias.shape)} dtype 保持 fp32")
print(f"      config.scoring_func = {json.load(open('/tmp/glm53f/config.json'))['text_config']['scoring_func']}"
      f"  → 源码 L161 用 router_logits.sigmoid()，非 softmax")
print(f"      config.topk_method = {json.load(open('/tmp/glm53f/config.json'))['text_config']['topk_method']}")
print(f"      n_routed_experts={C.n_routed_experts} num_experts_per_tok={C.num_experts_per_tok}")
print(f"      n_group={C.n_group} topk_group={C.topk_group} norm_topk_prob={C.norm_topk_prob}")
print(f"      routed_scaling_factor={C.routed_scaling_factor}")
print(f"      router_logits 在 fp32 计算（L160 两侧都 .type(torch.float32)）；"
      f"config.moe_router_dtype={json.load(open('/tmp/glm53f/config.json'))['text_config']['moe_router_dtype']}")

# ---------- 4.2 路由输出实测 ----------
banner("4.2 路由输出实测（真实 288 专家 / topk 8）")
with torch.no_grad():
    router.weight.normal_(0, 0.02)
    router.e_score_correction_bias.normal_(0, 0.1)   # 训练后为非零值，随机模拟
x = torch.randn(1, 8, C.hidden_size, dtype=DT)
with torch.no_grad():
    logits, w, idx = router(x)
print(f"      router_logits {tuple(logits.shape)} dtype={logits.dtype}")
print(f"      topk_indices  {tuple(idx.shape)}  取值范围 [{int(idx.min())}, {int(idx.max())}]  "
      f"每行是否互不重复={bool(all(len(set(r.tolist()))==C.num_experts_per_tok for r in idx))}")
print(f"      topk_weights  {tuple(w.shape)}  范围 [{w.min():.6f}, {w.max():.6f}]")
print(f"      每 token 权重之和 = {w.sum(-1).tolist()}")
print(f"      → norm_topk_prob 归一化后和为 1，再乘 routed_scaling_factor={C.routed_scaling_factor}，"
      f"故和恒为 {C.routed_scaling_factor}")
print(f"      归一化前(即除以 scaling 后)之和 = {(w.sum(-1)/C.routed_scaling_factor).tolist()}")

# ---------- 4.3 手工复算路由，核对公式 ----------
banner("4.3 手算复现路由公式（逐步核对）")
with torch.no_grad():
    hs = x.view(-1, C.hidden_size)
    rl = F.linear(hs.float(), router.weight.float())
    scores = rl.sigmoid()                                    # 步1 sigmoid
    sfc = scores + router.e_score_correction_bias            # 步2 加偏置（只影响选择）
    # n_group=1 → 分组退化：group_scores 只有一组，group_mask 全 1
    gs = sfc.view(-1, C.n_group, C.n_routed_experts // C.n_group).topk(2, dim=-1)[0].sum(-1)
    gi = torch.topk(gs, k=C.topk_group, dim=-1, sorted=False)[1]
    gm = torch.zeros_like(gs); gm.scatter_(1, gi, 1)
    sm = gm.unsqueeze(-1).expand(-1, C.n_group, C.n_routed_experts // C.n_group).reshape(-1, C.n_routed_experts)
    print(f"      n_group=1 → score_mask 是否全为 1 = {bool(sm.all())}（分组机制退化为全局 topk）")
    sfc2 = sfc.masked_fill(~sm.bool(), float("-inf"))
    ti = torch.topk(sfc2, k=C.num_experts_per_tok, dim=-1, sorted=False)[1]
    tw = scores.gather(1, ti)                                # 步3 权重取原始 score，不含偏置
    tw = tw / (tw.sum(-1, keepdim=True) + 1e-20)             # 步4 归一化
    tw = tw * C.routed_scaling_factor                        # 步5 缩放
print(f"      手算 indices 与官方一致 = {bool((torch.sort(ti,-1)[0]==torch.sort(idx,-1)[0]).all())}")
print(f"      手算 weights 与官方最大误差 = {float((torch.sort(tw,-1)[0]-torch.sort(w,-1)[0]).abs().max()):.3e}")
print(f"      对照：直接全局 topk(sfc) 的结果与官方一致 = "
      f"{bool((torch.sort(torch.topk(sfc,C.num_experts_per_tok,-1)[1],-1)[0]==torch.sort(idx,-1)[0]).all())}")

# ---------- 4.4 偏置只影响选择不影响权重 ----------
banner("4.4 e_score_correction_bias 的作用范围（aux-loss-free 路由的关键）")
with torch.no_grad():
    r2 = Glm5NextTextTopkRouter(C).to(DT).eval()
    r2.weight.copy_(router.weight)
    r2.e_score_correction_bias.zero_()
    _, w0, idx0 = r2(x)
    # 用官方带偏置的选择索引，从无偏置路由器取权重
    s_noB = F.linear(x.view(-1, C.hidden_size).float(), router.weight.float()).sigmoid()
changed = int((torch.sort(idx,-1)[0] != torch.sort(idx0,-1)[0]).any(-1).sum())
print(f"      加偏置 vs 不加偏置：{changed}/{idx.shape[0]} 个 token 的专家选择发生变化")
w_from_official_idx = s_noB.gather(1, idx)
w_norm = w_from_official_idx / w_from_official_idx.sum(-1, keepdim=True) * C.routed_scaling_factor
print(f"      按官方(带偏置)选出的索引、用不含偏置的 score 算权重，与官方权重误差 = "
      f"{float((w_norm - w).abs().max()):.3e}")
print(f"      → 偏置进入 scores_for_choice（L162）用于 topk；权重走 scores.gather（L178）不含偏置")
print(f"      这是 DeepSeek noaux_tc 的做法：靠偏置调负载均衡，不污染专家输出权重，也不需要辅助损失")
print(f"      config.router_aux_loss_coef = {C.router_aux_loss_coef}（仅在 output_router_logits 时用于训练侧统计）")

# ---------- 4.5 SwiGLU clamp ----------
banner("4.5 SwiGLU 的 clamp（swiglu_limit=10.0）")
print(f"      config.swiglu_limit = {C.swiglu_limit}")
print(f"      源码 L102-103 / L139-140：gate.clamp(min=None, max=limit)  up.clamp(min=-limit, max=limit)")
print(f"      → gate 只截上界（负半轴不截，silu 在负半轴本就趋 0）；up 双侧截断")
probe = torch.tensor([[-50.0, -10.0, 0.0, 5.0, 10.0, 50.0]], dtype=DT)
g_c = probe.clamp(min=None, max=C.swiglu_limit)
u_c = probe.clamp(min=-C.swiglu_limit, max=C.swiglu_limit)
print(f"      输入      {probe.tolist()[0]}")
print(f"      gate 后   {g_c.tolist()[0]}")
print(f"      up 后     {u_c.tolist()[0]}")
print(f"      silu(gate)*up = {(F.silu(g_c)*u_c).tolist()[0]}")

# ---------- 4.6 MoE 整体前向（缩小专家数以便实跑） ----------
banner("4.6 MoE 层完整前向（专家数缩至 16 以便本机实跑，其余结构不变）")
Cs = shrink_config(num_layers=8)
moe = Glm5NextTextMoE(Cs).to(DT).eval()
with torch.no_grad():
    for p in moe.parameters():
        p.normal_(0, 0.02)
xm = torch.randn(1, 6, Cs.hidden_size, dtype=DT)
with torch.no_grad():
    ym = moe(xm)
print(f"      输入 {tuple(xm.shape)} → 输出 {tuple(ym.shape)}  finite={bool(torch.isfinite(ym).all())}")
print(f"      experts.gate_up_proj {tuple(moe.experts.gate_up_proj.shape)} "
      f"= (E, 2*moe_inter, hidden)  ← gate 与 up 打包在一个张量")
print(f"      experts.down_proj    {tuple(moe.experts.down_proj.shape)} = (E, hidden, moe_inter)")
print(f"      shared_experts 的 intermediate = moe_intermediate_size * n_shared_experts = "
      f"{Cs.moe_intermediate_size}*{Cs.n_shared_experts} = {moe.shared_experts.intermediate_size}")
print(f"      MoE.forward（L200-207）：routed 输出 + shared_experts(原始输入) —— 共享专家对所有 token 都算")

# 分解验证：routed 与 shared 两路
with torch.no_grad():
    _, tw2, ti2 = moe.gate(xm)
    routed_only = moe.experts(xm.view(-1, Cs.hidden_size), ti2, tw2).view(xm.shape)
    shared_only = moe.shared_experts(xm)
print(f"      routed 分量范数={float(routed_only.norm()):.4f}  shared 分量范数={float(shared_only.norm()):.4f}")
print(f"      routed+shared 与 MoE 输出误差 = {float((routed_only+shared_only-ym).abs().max()):.3e}")

# ---------- 4.7 每 token 激活参数量 ----------
banner("4.7 每 token 激活参数量（真实 config，按张量形状精确统计）")
Dh, Mi, Ii = C.hidden_size, C.moe_intermediate_size, C.intermediate_size
one_expert = 3 * Dh * Mi
shared = 3 * Dh * Mi * C.n_shared_experts
router_p = C.n_routed_experts * Dh
moe_act = C.num_experts_per_tok * one_expert + shared + router_p
dense_mlp = 3 * Dh * Ii
print(f"      单个 routed 专家 = 3*hidden*moe_inter = 3*{Dh}*{Mi} = {one_expert:,}")
print(f"      激活 {C.num_experts_per_tok} 个 = {C.num_experts_per_tok*one_expert:,}")
print(f"      共享专家 = {shared:,}   路由器 = {router_p:,}")
print(f"      → MoE 层每 token 激活 = {moe_act:,} = {moe_act/1e6:.2f} M")
print(f"      dense MLP 层 = 3*hidden*inter = 3*{Dh}*{Ii} = {dense_mlp:,} = {dense_mlp/1e6:.2f} M")
kda_p, dsa_p, hc_p = 137_732_288, 124_914_432, 2*393_243   # 来自探针1/探针3的实测
n_kda = sum(1 for t in C.layer_types if t == "linear_attention")
n_dsa = sum(1 for t in C.layer_types if t == "deepseek_sparse_attention")
n_dense = sum(1 for t in C.mlp_layer_types if t == "dense")
n_sparse = sum(1 for t in C.mlp_layer_types if t == "sparse")
act = (n_kda*kda_p + n_dsa*dsa_p + 45*hc_p + n_dense*dense_mlp + n_sparse*moe_act
       + 2*C.vocab_size*C.hidden_size)
print(f"\n      分项累计（KDA {n_kda} 层 / DSA {n_dsa} 层 / dense {n_dense} / sparse {n_sparse}）：")
print(f"        KDA 注意力   {n_kda*kda_p:>15,}")
print(f"        DSA 注意力   {n_dsa*dsa_p:>15,}")
print(f"        mHC          {45*hc_p:>15,}")
print(f"        dense MLP    {n_dense*dense_mlp:>15,}")
print(f"        MoE 激活     {n_sparse*moe_act:>15,}")
print(f"        embed+lm_head{2*C.vocab_size*C.hidden_size:>15,}")
print(f"      合计每 token 激活 ≈ {act:,} = {act/1e9:.2f} B")
print(f"      官方 README 声称「320B total / 18B active」；checkpoint 实测总量 321.32 B")
print(f"      本探针分项累计 {act/1e9:.2f} B（含 embed+lm_head），与官方 18B 口径吻合到 "
      f"{abs(act/1e9-18)/18*100:.1f}% 以内；扣除 embed+lm_head 为 "
      f"{(act-2*C.vocab_size*C.hidden_size)/1e9:.2f} B。")
print(f"      MTP 层（layer 45，{7_432_592_416/1e9:.2f} B）不计入上述激活量：投机解码时才前向。")
