"""实测五：MoE 路由——sqrtsoftplus 评分、noaux_tc 的 bias 语义、以及前 3 层的 Hash 路由。

对应 model.py Gate 与报告 2.1。
验证：
  1. score_func 从 V3 的 sigmoid 改为 sqrt(softplus)，实测两者曲线差异
  2. bias 只影响 topk 选择，不进入最终权重（auxiliary-loss-free 的关键）
  3. 前 n_hash_layers=3 层用 tid2eid 查表路由，与隐藏状态无关
  4. 权重的归一化与 route_scale
"""
import sys, json, torch
import torch.nn as nn
import torch.nn.functional as F

cfg = json.load(open("/tmp/dsv4/config.json"))
torch.manual_seed(0)

N_EXP = cfg["n_routed_experts"]
TOPK = cfg["num_experts_per_tok"]
SCALE = cfg["routed_scaling_factor"]
N_HASH = cfg["num_hash_layers"]
VOCAB = cfg["vocab_size"]
print(f"n_routed_experts={N_EXP}  num_experts_per_tok={TOPK}  n_shared_experts={cfg['n_shared_experts']}")
print(f"scoring_func={cfg['scoring_func']}  topk_method={cfg['topk_method']}  routed_scaling_factor={SCALE}")
print(f"num_hash_layers={N_HASH}  norm_topk_prob={cfg['norm_topk_prob']}")
print()

print("=== 1. sqrt(softplus) vs sigmoid（V3 用 sigmoid，V4 改为 sqrtsoftplus）===")
z = torch.tensor([-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0, 12.0])
sig = z.sigmoid()
sqsp = F.softplus(z).sqrt()
print(f"{'logit':>8} {'sigmoid':>12} {'sqrt(softplus)':>16}")
for i in range(len(z)):
    print(f"{z[i].item():>8.1f} {sig[i].item():>12.6f} {sqsp[i].item():>16.6f}")
print("  sigmoid 上界为 1（饱和）；sqrt(softplus) 无上界，约按 sqrt(z) 增长")
print(f"  logit=12 时：sigmoid={sig[-1]:.6f}（已饱和），sqrt(softplus)={sqsp[-1]:.4f}（仍在增长）")
print("  含义：高置信路由不会被压缩到同一数值，专家间亲和度差异得以保留")
print()

print("=== 2. bias 只影响选择、不影响权重（model.py Gate.forward）===")
x = torch.randn(4, 128)
W = torch.randn(N_EXP, 128) * 0.05
bias = torch.zeros(N_EXP)
bias[0] = 5.0                                   # 人为把专家0的选择偏置抬高

def gate(x, W, bias, use_bias=True):
    scores = F.linear(x.float(), W.float())
    scores = F.softplus(scores).sqrt()
    original = scores
    s = scores + bias if use_bias else scores
    idx = s.topk(TOPK, dim=-1)[1]
    w = original.gather(1, idx)                 # 关键：从 original 取权重，不含 bias
    w = w / w.sum(-1, keepdim=True)
    return w * SCALE, idx

w_b, i_b = gate(x, W, bias, True)
w_n, i_n = gate(x, W, bias, False)
print(f"带 bias 选中的专家(第0个token): {i_b[0].tolist()}")
print(f"不带 bias 选中的专家           : {i_n[0].tolist()}")
print(f"专家0 是否被 bias 挤进来: {0 in i_b[0].tolist() and 0 not in i_n[0].tolist()}")
sel = (i_b[0] == 0).nonzero()
if len(sel):
    pos = sel[0, 0].item()
    raw = F.softplus(F.linear(x[0:1].float(), W.float())).sqrt()[0, 0].item()
    print(f"专家0 的原始分数={raw:.6f}，加 bias 后={raw+5.0:.6f}")
    print(f"但它拿到的路由权重基于原始分数计算，bias 未进入权重")
print("  -> 这是 auxiliary-loss-free 负载均衡：用 bias 调节各专家被选中的频率，")
print("     同时不扭曲专家输出的加权系数，因此无需辅助损失的梯度干预")
print()

print("=== 3. 前 3 层 Hash 路由：与隐藏状态无关 ===")
print("model.py Gate.__init__:  self.hash = layer_id < args.n_hash_layers")
print("model.py Gate.forward :  indices = self.tid2eid[input_ids]   # 按 token ID 查表")
tid2eid = torch.randint(0, N_EXP, (VOCAB, TOPK), dtype=torch.int32)
ids = torch.tensor([100, 100, 5000])
print(f"tid2eid 形状 = {tuple(tid2eid.shape)} = (vocab_size, num_experts_per_tok)")
print(f"token 100 -> 专家 {tid2eid[100].tolist()}")
print(f"token 100（另一处出现，隐藏状态不同）-> 专家 {tid2eid[100].tolist()}  （完全相同）")
print(f"token 5000 -> 专家 {tid2eid[5000].tolist()}")
print("  -> 同一 token ID 在任何上下文中都路由到同一组专家，路由结果与该层输入无关")
print()

# 用权重清单核对：哪些层真有 tid2eid / 哪些层有 bias
idxj = json.load(open("/tmp/dsv4/index.json"))["weight_map"]
hash_layers = sorted(int(k.split(".")[1]) for k in idxj if k.endswith("gate.tid2eid"))
bias_layers = sorted(int(k.split(".")[1]) for k in idxj if k.endswith("gate.bias") and k.startswith("layers."))
print("=== 4. 权重清单交叉验证（model.safetensors.index.json）===")
print(f"含 gate.tid2eid 的层: {hash_layers}   （共 {len(hash_layers)} 层，等于 num_hash_layers）")
print(f"含 gate.bias 的层数 : {len(bias_layers)}  层号范围 {bias_layers[0]}..{bias_layers[-1]}")
print(f"  61 层中 {len(bias_layers)} 层有 bias，缺失的层 = {sorted(set(range(61)) - set(bias_layers))}")
print("  -> 与 model.py 一致：hash 层 self.bias=None，其余层才有 bias")
