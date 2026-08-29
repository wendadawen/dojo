"""实测 3：MoE 路由（softmax→topk→重归一化 + aux loss）与共享专家门控。

验证目标（对应源码 qwen3_vl_moe Qwen3VLMoeTextTopKRouter + qwen3_next Qwen3NextSparseMoeBlock）：
  3A. 路由权重经 softmax→topk→top-k 内重归一化，每 token 权重和 = 1
  3B. 对比不重归一化（原始 softmax 概率）的权重差异
  3C. aux loss 负载均衡公式（router_aux_loss_coef=0.001）实算
  3D. 共享专家 sigmoid 门：逐 token 标量缩放
  3E. 专家计算：gate_up chunk → silu(gate)*up → down，加权和
真实维度：512 专家 top-10，moe_intermediate 1024，hidden 4096。
"""
import torch, torch.nn.functional as F

torch.manual_seed(2)
E, TOPK, I, HID = 512, 10, 1024, 4096
INIT = 0.02   # config initializer_range=0.02；路由器初始化为全零，训练后为小量级

# ============ 3A. 路由归一化 ============
print("=== 3A. softmax→topk→重归一化 ===")
Wg = torch.randn(E, HID) * INIT
x = torch.randn(6, HID)
logits = x @ Wg.T
probs = F.softmax(logits, dim=-1, dtype=torch.float32)
tv, ti = torch.topk(probs, TOPK, dim=-1)
w_renorm = tv / tv.sum(dim=-1, keepdim=True)
print(f"  top-{TOPK} 重归一化后每 token 权重和: {[f'{v:.6f}' for v in w_renorm.sum(-1).tolist()]}")
print(f"  → 重归一化是无条件的（Qwen3.5 的路由器没有 norm_topk_prob 开关）")

# ============ 3B. 对比原始概率 ============
print()
print("=== 3B. 重归一化 vs 原始 softmax 概率 ===")
top1_share_renorm = w_renorm[:, 0]
top1_share_raw = tv[:, 0]
print(f"  原始 top-1 概率:   {[f'{v:.4f}' for v in top1_share_raw.tolist()]}")
print(f"  重归一 top-1 权重: {[f'{v:.4f}' for v in top1_share_renorm.tolist()]}")
raw_sum = tv.sum(-1)
print(f"  top-{TOPK} 原始概率合计: {[f'{v:.4f}' for v in raw_sum.tolist()]}")
print(f"  → 512 专家下 top-10 原始概率常不足 0.5，重归一化把量级放大 {1/raw_sum.mean():.2f}×")

# ============ 3C. aux loss ============
print()
print("=== 3C. 辅助损失（负载均衡）实算 ===")
# transformers 的 load_balancing_loss_func：freq_i = 每 token 上专家 i 入选比例；prob_i = softmax 概率平均
# loss = E * sum_i(freq_i * prob_i)，再加系数
sel = F.one_hot(ti, num_classes=E).sum(1).float()      # [T, E] 每专家被选次数
freq = sel / sel.sum(-1, keepdim=True) / TOPK * TOPK   # 每 token 平均
prob_mean = probs.mean(0)
aux = E * (freq * prob_mean).sum() * 0.001
perfect = E * (torch.full((E,), 1/E) * torch.full((E,), 1/E)).sum() * 0.001
print(f"  随机路由下 6 token 的 aux loss = {aux.item():.6f}（coef=0.001 已乘）")
print(f"  完全均匀分布的下界参考 = {perfect.item():.6f}")
print(f"  → 路由用 softmax 概率（非 sigmoid 独立打分），均衡靠 aux loss 梯度而非推理期 bias 修正")

# ============ 3D. 共享专家门 ============
print()
print("=== 3D. 共享专家 sigmoid 门 ===")
Wsg = torch.randn(1, HID) * INIT
shared_out = torch.randn(6, HID)
gate = torch.sigmoid(x @ Wsg.T)                        # [T,1] 逐 token 标量
print(f"  逐 token 门值: {[f'{v[0]:.4f}' for v in gate.tolist()]}")
print(f"  共享专家恒激活（不参与路由），输出乘 sigmoid(scalar) 后与路由专家相加")
print(f"  → 三个门在一条 FFN 通路：路由权重（向量）、共享专家门（标量）、注意力输出门（逐头向量）")

# ============ 3E. 专家前向 ============
print()
print("=== 3E. 单个专家前向（SwiGLU） ===")
gu = torch.randn(2 * I, HID)
dn = torch.randn(HID, I)
h = torch.randn(1, HID)
gate_p, up_p = (h @ gu.T).chunk(2, dim=-1)
expert_out = (F.silu(gate_p) * up_p) @ dn.T
print(f"  gate_up_proj [2×{I}, {HID}] chunk 成 gate/up 各 [{I}]，silu(gate)*up 后 down [{HID},{I}]")
print(f"  单专家激活参数 = {2*I*HID + HID*I:,} = {2*I*HID + HID*I:.0f}")
print(f"  路由专家每 token 激活 {TOPK} 个 = {TOPK*(2*I*HID + HID*I):,}，共享专家恒激活 = {3*I*HID:,}")
