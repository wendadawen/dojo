"""实测八：给官方 model.py 打钩子，抓取真实前向中每一级张量形状与候选集规模。

不改架构代码，用 forward hook / 包装函数记录实际张量。
产出用于数据流图的确切数字。
"""
import sys, types, json
import torch

sys.path.insert(0, "/tmp/dsv4/exp")
import kernel_ref

fake = types.ModuleType("kernel")
fake.act_quant = kernel_ref.act_quant
fake.fp4_act_quant = kernel_ref.fp4_act_quant
fake.sparse_attn = kernel_ref.sparse_attn
fake.hc_split_sinkhorn = kernel_ref.hc_split_sinkhorn
fake.fp8_gemm = kernel_ref.fp8_gemm
fake.fp4_gemm = kernel_ref.fp4_gemm
sys.modules["kernel"] = fake
sys.path.insert(0, "/tmp/dsv4/inference")
import model as M

def hadamard(x):
    n = x.size(-1)
    y = x.clone().float(); h = 1
    while h < n:
        y = y.unflatten(-1, (-1, 2, h))
        a, b = y[..., 0, :].clone(), y[..., 1, :].clone()
        y = torch.stack([a + b, a - b], dim=-2).flatten(-3); h *= 2
    return (y * n ** -0.5).to(x.dtype)
M.rotate_activation = hadamard

torch.manual_seed(0)
torch.set_default_dtype(torch.bfloat16)
cfg = json.load(open("/tmp/dsv4/config.json"))

args = M.ModelArgs(
    max_batch_size=1, max_seq_len=1024, dtype="bf16", scale_fmt=None,
    expert_dtype=None, scale_dtype="fp32",
    vocab_size=512, dim=256, moe_inter_dim=128,
    n_layers=4, n_hash_layers=1, n_mtp_layers=1,
    n_heads=8, n_routed_experts=8, n_shared_experts=1, n_activated_experts=2,
    score_func="sqrtsoftplus", route_scale=2.5, swiglu_limit=10.0,
    q_lora_rank=128, head_dim=128, rope_head_dim=64, o_groups=4, o_lora_rank=64,
    window_size=128, compress_ratios=(128, 4, 128, 4, 0),
    compress_rope_theta=160000, original_seq_len=65536,
    rope_theta=10000, rope_factor=16, beta_fast=32, beta_slow=1,
    index_n_heads=8, index_head_dim=64, index_topk=16,
    hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
)
model = M.Transformer(args)
for p in model.parameters():
    if p.dtype.is_floating_point:
        with torch.no_grad():
            p.normal_(0, 0.02) if p.dim() > 1 else p.fill_(0.02)
for layer in model.layers:
    if getattr(layer.ffn.gate, "hash", False):
        with torch.no_grad():
            layer.ffn.gate.tid2eid.copy_(torch.randint(0, 8, layer.ffn.gate.tid2eid.shape, dtype=torch.int32))
for m in model.modules():
    if isinstance(m, M.RMSNorm):
        with torch.no_grad(): m.weight.fill_(1.0)

log = []
# 抓 sparse_attn 的实际候选集规模
orig_sa = kernel_ref.sparse_attn
def traced_sa(q, kv, sink, idxs, scale):
    valid = (idxs != -1).sum(-1)
    log.append(("sparse_attn", tuple(q.shape), tuple(kv.shape), tuple(idxs.shape),
                int(valid.max()), int(valid.float().mean())))
    return orig_sa(q, kv, sink, idxs, scale)
fake.sparse_attn = traced_sa
M.sparse_attn = traced_sa

shapes = {}
def mk_hook(name):
    def hook(mod, inp, out):
        def sh(t):
            return tuple(t.shape) if torch.is_tensor(t) else None
        shapes.setdefault(name, (sh(inp[0]) if inp and torch.is_tensor(inp[0]) else None,
                                 sh(out) if torch.is_tensor(out) else None))
    return hook

model.embed.register_forward_hook(mk_hook("embed"))
for i, layer in enumerate(model.layers):
    layer.attn.register_forward_hook(mk_hook(f"L{i}.attn"))
    layer.attn.wq_a.register_forward_hook(mk_hook(f"L{i}.attn.wq_a"))
    layer.attn.wq_b.register_forward_hook(mk_hook(f"L{i}.attn.wq_b"))
    layer.attn.wkv.register_forward_hook(mk_hook(f"L{i}.attn.wkv"))
    layer.attn.wo_b.register_forward_hook(mk_hook(f"L{i}.attn.wo_b"))
    if layer.attn.compress_ratio:
        layer.attn.compressor.register_forward_hook(mk_hook(f"L{i}.attn.compressor"))
        if layer.attn.indexer is not None:
            layer.attn.indexer.register_forward_hook(mk_hook(f"L{i}.attn.indexer"))
    layer.ffn.register_forward_hook(mk_hook(f"L{i}.ffn"))
    layer.ffn.gate.register_forward_hook(mk_hook(f"L{i}.ffn.gate"))
    layer.register_forward_hook(mk_hook(f"L{i}.block"))

SEQ = 512
ids = torch.randint(0, args.vocab_size, (1, SEQ))
print(f"=== prefill seqlen={SEQ}，dim={args.dim}, head_dim={args.head_dim}, n_heads={args.n_heads} ===")
out = model(ids, 0)
print()
print("--- 各级张量形状（forward hook 实测）---")
for k in ["embed"] + sum([[f"L{i}.{s}" for s in
          ["attn.wq_a","attn.wq_b","attn.wkv","attn.compressor","attn.indexer","attn.wo_b","attn","ffn.gate","ffn","block"]]
          for i in range(args.n_layers)], []):
    if k in shapes:
        i_, o_ = shapes[k]
        print(f"  {k:<22} in={str(i_):<22} out={o_}")
print()
print("--- sparse_attn 实际候选集（每层一次调用）---")
print(f"{'层':>4} {'q形状':<22} {'kv形状':<20} {'topk_idxs':<20} {'最大有效候选':>12}")
for i, r in enumerate(log):
    _, qs, kvs, idxs, vmax, vmean = r
    print(f"{i:>4} {str(qs):<22} {str(kvs):<20} {str(idxs):<20} {vmax:>12}")
print()
print("对照：window_size=128, index_topk=16, ratio=4 时压缩条目=512/4=128 -> CSA 候选=128+min(16,128)=144")
print("      ratio=128 时压缩条目=512/128=4 -> HCA 候选=128+4=132")
print()
print(f"最终 logits: {tuple(out.shape)}")
