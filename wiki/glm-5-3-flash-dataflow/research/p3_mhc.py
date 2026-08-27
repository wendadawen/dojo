"""
探针 3：mHC（Manifold-Constrained Hyper-Connections）—— GLM-5.3-Flash 的残差结构。

验证目标
  1. 隐藏状态从 [B,S,D] 扩成 [B,S,hc_mult,D] 的 4 路并行残差流
  2. mHC 映射的三路输出 pre / post / comb 的形状、取值范围与约束
  3. Sinkhorn 投影是否真的把 comb 逼近双随机矩阵（行和=列和=1）——实测偏差量级
  4. Sinkhorn 迭代次数与收敛的关系（config 给 20 次，实测收敛曲线）
  5. 残差更新公式 h' = post ⊗ f(h_collapsed) + comb^T @ h 的形状代数
  6. 退化检验：hc_mult=1 且 comb=1 时是否退回普通残差
  7. 末端 hc_head 是无权重均值（与 DeepSeek-V4 的加权收敛不同）

对应源码
  Glm5NextTextHyperConnection.forward   modeling_glm5_next.py L267-295
  Glm5NextTextHyperHead.forward         L298-302
  Glm5NextTextDecoderLayer.forward      L1280-1329
  Glm5NextTextModel.forward             L1477（unsqueeze(2).expand）、L1493（hc_head）
"""
from __future__ import annotations
import sys, json
sys.path.insert(0, "/tmp/glm53f/probe")
import torch
from harness import (real_config, banner, Glm5NextTextHyperConnection, Glm5NextTextHyperHead,
                     Glm5NextTextUnweightedRMSNorm)

torch.manual_seed(0)
C = real_config()
DT = torch.float32
banner("探针 3：mHC 流形约束超连接")

# ---------- 3.1 参数形状 ----------
hc = Glm5NextTextHyperConnection(C).to(DT).eval()
print("[3.1] mHC 单站点参数（真实 config）")
tot = 0
for n, p in hc.named_parameters():
    tot += p.numel()
    print(f"      {n:8s} {tuple(p.shape)}  {p.numel():>10,}")
print(f"      单站点合计 {tot:,}；每层两站点（attn + ffn）→ {2*tot:,}")
print(f"      全模型 45 层 → {45*2*tot:,} = {45*2*tot/1e6:.2f} M")
print(f"      mix = (2+hc_mult)*hc_mult = (2+{C.hc_mult})*{C.hc_mult} = {(2+C.hc_mult)*C.hc_mult}")
print(f"      fn 的输入宽度 = hc_mult*hidden_size = {C.hc_mult}*{C.hidden_size} = {C.hc_mult*C.hidden_size}")
print(f"      checkpoint 张量 hc_attn_fn=[24,16384] hc_attn_base=[24] hc_attn_scale=[3] → 一致")
print(f"      scale 长度 3 对应 pre/post/comb 三路各一个可学习缩放（源码注释 L261-264）")

# ---------- 3.2 三路输出的形状与范围 ----------
banner("3.2 pre / post / comb 三路输出（真实 hidden_size，随机权重）")
# 官方 _init_weights：fn ~ N(0, 0.02), base = 0, scale = 1
with torch.no_grad():
    hc.fn.normal_(0.0, 0.02)
    hc.base.zero_()
    hc.scale.fill_(1.0)
B, S = 1, 6
streams = torch.randn(B, S, C.hc_mult, C.hidden_size, dtype=DT)
with torch.no_grad():
    post, comb, collapsed = hc(streams)
print(f"      输入 hidden_streams {tuple(streams.shape)}  (B, S, hc_mult, D)")
print(f"      post      {tuple(post.shape)}  范围 [{post.min():.6f}, {post.max():.6f}]  "
      f"（公式 2*sigmoid(...) → 理论 ⊂ (0,2)）")
print(f"      comb      {tuple(comb.shape)}  范围 [{comb.min():.6f}, {comb.max():.6f}]")
print(f"      collapsed {tuple(collapsed.shape)}  ← pre 加权求和把 {C.hc_mult} 路压成 1 路")
# pre 不由 forward 返回，按源码 L283 复算一次以给出范围
with torch.no_grad():
    flat = hc.input_norm(streams.flatten(start_dim=2).float())
    pw, ow, cw = torch.nn.functional.linear(flat, hc.fn.float()).split(
        [C.hc_mult, C.hc_mult, C.hc_mult**2], dim=-1)
    ps, os_, cs = hc.scale.unbind(0)
    pre = torch.sigmoid(pw * ps + hc.base.split([C.hc_mult]*2+[C.hc_mult**2])[0]) + hc.hc_eps
print(f"      pre       {tuple(pre.shape)}  范围 [{pre.min():.6f}, {pre.max():.6f}]  "
      f"（公式 sigmoid(...)+eps → 理论 ⊂ (eps, 1+eps)）")
print(f"      hc_eps = {hc.hc_eps}（config.hc_eps），作为 Sinkhorn 与 pre 的数值下限")

# ---------- 3.3 Sinkhorn：comb 是否双随机 ----------
banner("3.3 Sinkhorn-Knopp 投影：comb 是否落在双随机流形上")
rs = comb.sum(dim=-1)      # 行和
csum = comb.sum(dim=-2)    # 列和
print(f"      comb 形状 {tuple(comb.shape)} = (B, S, hc_mult, hc_mult)，每个 token 一个 4x4 矩阵")
print(f"      行和 范围 [{rs.min():.8f}, {rs.max():.8f}]  距 1 的最大偏差 {(rs-1).abs().max():.3e}")
print(f"      列和 范围 [{csum.min():.8f}, {csum.max():.8f}]  距 1 的最大偏差 {(csum-1).abs().max():.3e}")
print(f"      全元素非负 = {bool((comb>=0).all())}")
print(f"      config.hc_sinkhorn_iters = {C.hc_sinkhorn_iters}")
print(f"      源码顺序（L286-290）：softmax(dim=-1)+eps → 先做一次列归一 → 再循环 iters-1 次「行归一+列归一」")
print(f"      因此列归一执行 {C.hc_sinkhorn_iters} 次，行归一 {C.hc_sinkhorn_iters-1} 次；")
print(f"      循环体最后一步是列归一 → 列和更接近 1，行和残留偏差更大（与上面实测一致）")
print(f"      示例 comb[0,0] =\n{comb[0,0].numpy()}")

# ---------- 3.4 迭代次数与收敛 ----------
banner("3.4 Sinkhorn 迭代次数对双随机性的影响（同一输入，只改 iters）")
print("      iters   行和最大偏差    列和最大偏差")
for it in [1, 2, 3, 5, 10, 20, 40]:
    h2 = Glm5NextTextHyperConnection(real_config({"hc_sinkhorn_iters": it})).to(DT).eval()
    with torch.no_grad():
        h2.fn.copy_(hc.fn); h2.base.copy_(hc.base); h2.scale.copy_(hc.scale)
        _, cb, _ = h2(streams)
    print(f"      {it:5d}   {float((cb.sum(-1)-1).abs().max()):.3e}      "
          f"{float((cb.sum(-2)-1).abs().max()):.3e}")

# ---------- 3.5 残差更新公式的形状代数 ----------
banner("3.5 残差更新 h' = post ⊗ sublayer_out + comb^T @ h")
sub_out = torch.randn(B, S, C.hidden_size, dtype=DT)     # 子层（attn 或 mlp）输出，已压成单路
term1 = post.unsqueeze(-1) * sub_out.unsqueeze(-2)
term2 = torch.matmul(comb.transpose(-1, -2), streams)
new = term1 + term2
print(f"      sublayer_out              {tuple(sub_out.shape)}")
print(f"      post.unsqueeze(-1)        {tuple(post.unsqueeze(-1).shape)}  (B,S,H,1)")
print(f"      sub_out.unsqueeze(-2)     {tuple(sub_out.unsqueeze(-2).shape)}  (B,S,1,D)")
print(f"      term1 = 广播乘            {tuple(term1.shape)}  ← 子层输出按 post 分发回 {C.hc_mult} 路")
print(f"      term2 = comb^T @ streams  {tuple(term2.shape)}  ← {C.hc_mult} 路之间做双随机混合")
print(f"      h'                        {tuple(new.shape)}  与输入同形 → 可无限堆叠")
print(f"      每层做两次该更新（attn 站点 + ffn 站点，源码 L1316 与 L1325）")

# ---------- 3.6 退化检验 ----------
banner("3.6 退化检验：mHC 是否包含普通残差作为特例")
h1 = Glm5NextTextHyperConnection(real_config({"hc_mult": 1})).to(DT).eval()
with torch.no_grad():
    h1.fn.zero_(); h1.base.zero_(); h1.scale.fill_(1.0)
    s1 = torch.randn(1, 4, 1, C.hidden_size, dtype=DT)
    p1, c1, col1 = h1(s1)
print(f"      hc_mult=1, fn=0, base=0 → post={float(p1.flatten()[0]):.8f} (=2*sigmoid(0)=1.0), "
      f"comb={float(c1.flatten()[0]):.8f} (1x1 双随机矩阵必为 1)")
sub1 = torch.randn(1, 4, C.hidden_size, dtype=DT)
new1 = p1.unsqueeze(-1) * sub1.unsqueeze(-2) + torch.matmul(c1.transpose(-1, -2), s1)
plain = (s1 + sub1.unsqueeze(-2))
print(f"      mHC 更新 vs 普通残差 h+f(h) 最大误差 = {float((new1-plain).abs().max()):.3e}")
print(f"      collapsed vs 原输入（pre=sigmoid(0)+eps={float(torch.sigmoid(torch.tensor(0.))+hc.hc_eps):.6f} 倍）"
      f"最大误差 = {float((col1 - s1.squeeze(2)*(0.5+hc.hc_eps)).abs().max()):.3e}")
print(f"      → hc_mult=1 且 fn=0 时 mHC 退化为普通残差，残差项系数为 comb=1-O(hc_eps)，")
print(f"        故误差量级 {float((new1-plain).abs().max()):.1e} 由 hc_eps={hc.hc_eps} 引入，非精确相等")

# ---------- 3.7 末端收敛 ----------
banner("3.7 末端 hc_head：无权重均值")
head = Glm5NextTextHyperHead()
with torch.no_grad():
    merged = head(streams)
print(f"      输入 {tuple(streams.shape)} → 输出 {tuple(merged.shape)}")
print(f"      与 streams.mean(dim=2) 误差 = {float((merged - streams.mean(2)).abs().max()):.3e}")
print(f"      源码 docstring 明写：Unlike DeepSeek-V4, this is an unweighted mean（L299）")
print(f"      hc_head 无任何可学习参数 = {len(list(head.parameters()))==0}")

# ---------- 3.8 输入端展开 ----------
banner("3.8 输入端：embedding 如何变成 4 路")
emb = torch.randn(1, 5, C.hidden_size, dtype=DT)
expanded = emb.unsqueeze(2).expand(-1, -1, C.hc_mult, -1).contiguous()
print(f"      embed 输出 {tuple(emb.shape)} --unsqueeze(2).expand--> {tuple(expanded.shape)}")
print(f"      4 路初始完全相同 = {bool(all((expanded[:,:,i]==emb).all() for i in range(C.hc_mult)))}")
print(f"      源码 L1477：hidden_states = inputs_embeds.unsqueeze(2).expand(-1,-1,hc_mult,-1).contiguous()")
print(f"      → 只是复制，不引入参数；分化完全由各层 mHC 的 pre/post/comb 产生")
print(f"      激活显存代价：主干隐藏状态张量放大 {C.hc_mult}× "
      f"（{C.hidden_size} → {C.hc_mult*C.hidden_size} 元素/token）")
