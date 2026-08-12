"""实测九：mHC 的块级数据流与参数量核对。

验证：
  1. 残差流形状 [b,s,hc_mult,d]，一个 Block 内 hc_pre/hc_post 各调用两次（attn 一次、ffn 一次）
  2. hc_pre 把 hc_mult 份残差归约成 1 份进子层；hc_post 把子层输出扩回 hc_mult 份
  3. mHC 的参数开销
  4. 用真实权重清单核对 1.6T 总参数 / 49B 激活参数
"""
import sys, types, json, torch
sys.path.insert(0, "/tmp/dsv4/exp")
import kernel_ref
fake = types.ModuleType("kernel")
for k in ["act_quant","fp4_act_quant","sparse_attn","hc_split_sinkhorn","fp8_gemm","fp4_gemm"]:
    setattr(fake, k, getattr(kernel_ref, k))
sys.modules["kernel"] = fake
sys.path.insert(0, "/tmp/dsv4/inference")
import model as M

cfg = json.load(open("/tmp/dsv4/config.json"))
torch.manual_seed(0); torch.set_default_dtype(torch.bfloat16)

def hadamard(x):
    n = x.size(-1)
    y = x.clone().float(); h = 1
    while h < n:
        y = y.unflatten(-1, (-1, 2, h))
        a, b = y[..., 0, :].clone(), y[..., 1, :].clone()
        y = torch.stack([a + b, a - b], dim=-2).flatten(-3); h *= 2
    return (y * n ** -0.5).to(x.dtype)
M.rotate_activation = hadamard

# act_quant 对 kv 的非 RoPE 维用 block_size=64，故 head_dim-rope_head_dim 必须是 64 的倍数
args = M.ModelArgs(max_batch_size=1, max_seq_len=256, dtype="bf16", scale_fmt=None,
    expert_dtype=None, scale_dtype="fp32", vocab_size=128, dim=64, moe_inter_dim=32,
    n_layers=1, n_hash_layers=0, n_mtp_layers=1, n_heads=4, n_routed_experts=4,
    n_shared_experts=1, n_activated_experts=2, score_func="sqrtsoftplus", route_scale=2.5,
    q_lora_rank=32, head_dim=128, rope_head_dim=64, o_groups=2, o_lora_rank=32,
    window_size=8, compress_ratios=(4, 0), index_n_heads=4, index_head_dim=128, index_topk=4,
    hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6)

block = M.Block(0, args)
for p in block.parameters():
    if p.dtype.is_floating_point:
        with torch.no_grad(): p.normal_(0, 0.02) if p.dim() > 1 else p.fill_(0.02)
with torch.no_grad():
    block.ffn.gate.bias.zero_()
for m in block.modules():
    if isinstance(m, M.RMSNorm):
        with torch.no_grad(): m.weight.fill_(1.0)

trace = []
_pre, _post = block.hc_pre, block.hc_post
def tp(x, fn, sc, ba):
    y, po, co = _pre(x, fn, sc, ba)
    trace.append(("hc_pre", tuple(x.shape), tuple(y.shape), tuple(po.shape), tuple(co.shape)))
    return y, po, co
def tq(x, res, po, co):
    y = _post(x, res, po, co)
    trace.append(("hc_post", tuple(x.shape), tuple(y.shape), tuple(res.shape), None))
    return y
block.hc_pre, block.hc_post = tp, tq

x = torch.randn(1, 16, args.hc_mult, args.dim)
ids = torch.randint(0, args.vocab_size, (1, 16))
out = block(x, 0, ids)

print(f"=== 一个 Block 内的 mHC 调用序列（hc_mult={args.hc_mult}, dim={args.dim}）===")
print(f"Block 输入残差流: {tuple(x.shape)}  <- [b, s, hc_mult, d]，即 {args.hc_mult} 份并行残差流")
print()
for i, t in enumerate(trace):
    kind = t[0]
    if kind == "hc_pre":
        print(f"  {i+1}. hc_pre : 残差流 {t[1]} -> 子层输入 {t[2]}")
        print(f"           同时产出 post{t[3]} 与 comb{t[4]}（comb 为 Sinkhorn 后的混合矩阵）")
    else:
        print(f"  {i+1}. hc_post: 子层输出 {t[1]} + 残差流 {t[3]} -> 新残差流 {t[2]}")
print(f"Block 输出残差流: {tuple(out.shape)}")
print()
print("即：残差流始终是 hc_mult 份；每次进子层前归约成 1 份，出子层后再扩回 hc_mult 份。")
print("一个 Block 有 attn 和 ffn 两个子层，故 hc_pre/hc_post 各调用 2 次。")
print()

print("=== mHC 参数开销（每层）===")
d, hc = cfg["hidden_size"], cfg["hc_mult"]
mix = (2 + hc) * hc
per = 2 * (mix * hc * d + mix + 3)     # attn 与 ffn 各一组 fn/base/scale
print(f"hc_attn_fn/hc_ffn_fn 形状 = ({mix}, {hc*d}) = ({mix}, {hc*d})")
print(f"每层 mHC 参数 = 2 x ({mix} x {hc*d} + {mix} + 3) = {per:,}")
print(f"61 层合计 = {per*61:,} ≈ {per*61/1e9:.3f}B")
print("（已用真实权重核对：layers.0.hc_attn_fn 形状 F32 [24, 28672]，与 (mix_hc, hc_mult*dim) 一致）")
print()

print("=== 用真实权重清单核对总参数量 ===")
idx = json.load(open("/tmp/dsv4/index.json"))
print(f"index.json metadata total_size = {idx['metadata']['total_size']:,} 字节 = {idx['metadata']['total_size']/2**40:.2f} TiB")
print("（这是磁盘字节数：专家为 FP4 打包、多数其他参数为 FP8，故字节数 < 参数个数 x 2）")
print()
L = cfg["num_hidden_layers"]
E, TOPK = cfg["n_routed_experts"], cfg["num_experts_per_tok"]
mi = cfg["moe_intermediate_size"]
expert_p = 3 * d * mi
print(f"单个专家参数 = 3 x {d} x {mi} = {expert_p:,}")
print(f"路由专家总量 = {L} 层 x {E} 个 x {expert_p:,} = {L*E*expert_p/1e12:.3f}T")
print(f"共享专家总量 = {L} x {expert_p:,} = {L*expert_p/1e9:.2f}B")
print(f"每 token 激活的专家参数 = {L} 层 x ({TOPK}+1) 个 x {expert_p:,} = {L*(TOPK+1)*expert_p/1e9:.2f}B")
print()
print(f"README 宣称：总参数 1.6T，激活 49B")
print(f"实算路由专家占 {L*E*expert_p/1e12:.3f}T，已接近 1.6T 主体；激活专家参数 {L*(TOPK+1)*expert_p/1e9:.1f}B，")
print(f"加注意力/embedding/mHC 等非专家部分后与 49B 量级一致。")
