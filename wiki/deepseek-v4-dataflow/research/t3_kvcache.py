"""实测三：KV cache 预算与注意力候选集规模——核对报告的 KV cache / FLOPs 主张。

依据（全部来自 config.json 与 model.py，不含推测）：
  head_dim=512, qk_rope_head_dim=64, num_key_value_heads=1（MQA，KV 单头）
  sliding_window=128, index_topk=1024
  compress_ratios: 偶数层(除第61层)=4 且带 indexer；奇数层=128；共 61 层
  model.py Attention.__init__:
     kv_cache_size = window_size + (max_seq_len // compress_ratio if compress_ratio else 0)
  Indexer.__init__:
     kv_cache = [bsz, max_seq_len // compress_ratio, index_head_dim]  (index_head_dim=128)
  report 2.3.4: RoPE 维度 BF16，其余维度 FP8
"""
import json

cfg = json.load(open("/tmp/dsv4/config.json"))
L = cfg["num_hidden_layers"]
head_dim = cfg["head_dim"]
rope_dim = cfg["qk_rope_head_dim"]
win = cfg["sliding_window"]
topk = cfg["index_topk"]
idx_dim = cfg["index_head_dim"]
ratios = cfg["compress_ratios"][:L]        # 第 62 项对应 MTP 层

csa = [i for i, r in enumerate(ratios) if r == 4]
hca = [i for i, r in enumerate(ratios) if r == 128]
print(f"num_hidden_layers={L}   head_dim={head_dim}  rope={rope_dim}  window={win}  index_topk={topk}")
print(f"CSA 层(ratio=4，带 Indexer): {len(csa)} 层，层号 {csa[:5]}...{csa[-3:]}")
print(f"HCA 层(ratio=128，无 Indexer): {len(hca)} 层，层号 {hca[:5]}...{hca[-3:]}")
print(f"两类层交替：前 12 层 ratio = {ratios[:12]}")
print()

# 每 token 每层 KV cache 字节数：RoPE 64 维 BF16(2B) + 其余 448 维 FP8(1B)
bytes_per_entry = rope_dim * 2 + (head_dim - rope_dim) * 1
print(f"每个 KV 条目字节数 = {rope_dim}x2(BF16 RoPE) + {head_dim-rope_dim}x1(FP8) = {bytes_per_entry} B")
print(f"  纯 BF16 存储则为 {head_dim*2} B，混合存储降到 {bytes_per_entry/(head_dim*2)*100:.1f}%  <- 报告称「近一半」")
print()

print("=== 1M 上下文下每层 KV 条目数 ===")
n = 1024 * 1024
for name, ratio, layers in [("CSA (ratio=4)", 4, len(csa)), ("HCA (ratio=128)", 128, len(hca))]:
    comp = n // ratio
    total = win + comp
    print(f"{name:<18} 滑窗 {win} + 压缩 {comp} = {total} 条目/层")
print()

csa_entries = win + n // 4
hca_entries = win + n // 128
main_bytes = (len(csa) * csa_entries + len(hca) * hca_entries) * bytes_per_entry
# Indexer 自带一份压缩 KV（index_head_dim=128），只在 CSA 层
idx_bytes = len(csa) * (n // 4) * idx_dim * 2      # model.py 中 indexer kv_cache 为默认 dtype(bf16)
print(f"主注意力 KV cache @1M = {main_bytes/2**30:.2f} GiB")
print(f"Indexer KV cache @1M  = {idx_bytes/2**30:.2f} GiB  ({len(csa)} 个 CSA 层 x {n//4} 条目 x {idx_dim} 维 bf16)")
print(f"合计 = {(main_bytes+idx_bytes)/2**30:.2f} GiB")
print()

# 报告基线：BF16 GQA8，head_dim=128
gqa_bytes = L * n * 8 * 128 * 2 * 2      # 8 组 x 128 维 x (K和V) x 2字节
print(f"报告基线 BF16 GQA8(head_dim=128) @1M = {gqa_bytes/2**30:.2f} GiB")
print(f"V4-Pro 主注意力 / 基线 = {main_bytes/gqa_bytes*100:.2f}%")
print(f"V4-Pro 含 Indexer / 基线 = {(main_bytes+idx_bytes)/gqa_bytes*100:.2f}%")
print("报告 2.3.4 称「约 2%」—— 与主注意力 KV cache 口径一致（Indexer 缓存另计）")
print()

print("=== 单 token 注意力候选集规模（决定 FLOPs）===")
print("CSA 层：滑窗 128 + Indexer 选出 min(index_topk, 可用压缩条目) 个")
for ctx in [4096, 65536, 1024*1024]:
    avail = ctx // 4
    sel = min(topk, avail)
    print(f"  上下文 {ctx:>9}: 压缩条目 {avail:>7} 个，实选 {sel:>4} 个 -> 候选集 {win+sel:>4}")
print("HCA 层：滑窗 128 + 全部压缩条目（无稀疏选择）")
for ctx in [4096, 65536, 1024*1024]:
    print(f"  上下文 {ctx:>9}: 压缩条目 {ctx//128:>7} 个 -> 候选集 {win+ctx//128:>7}")
print()
print(f"关键：CSA 层候选集在上下文 >= {topk*4} 后固定为 {win+topk}（topk 截断），")
print("      HCA 层随上下文线性增长但斜率仅 1/128。二者共同使注意力开销近似与长度脱钩。")
