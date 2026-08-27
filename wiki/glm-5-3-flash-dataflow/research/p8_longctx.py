"""
探针 8：混合架构在长上下文下的实际收益（全部数字由本脚本按真实 config 计算得出）。

验证目标
  1. 逐层 KV/状态占用：34 个 KDA 层是常数，11 个 DSA 层随长度线性增长
  2. 与"假设 45 层全是 MLA"的假想基线对比，算出真实节省比例
  3. DSA 的注意力打分次数：k-pool 把打分量从 O(S) 降到 O(S/kpool)，
     且选中池数有 512 的硬上限 → 长序列下命中率
  4. 45 层全 DSA / 45 层全 KDA / 真实混合 三种配置的对比表

所有口径都写清楚：KV 用 MLA 潜向量口径（kv_lora_rank，推理框架的实现方式），
KDA 用循环状态 + conv 状态。dtype 假设写在输出里。
"""
from __future__ import annotations
import json

CFG = json.load(open("/tmp/glm53f/config.json"))
T = CFG["text_config"]


def banner(t):
    print("=" * 78); print(t); print("=" * 78)


L = T["num_hidden_layers"]
n_kda = sum(1 for t in T["layer_types"] if t == "linear_attention")
n_dsa = sum(1 for t in T["layer_types"] if t == "deepseek_sparse_attention")
Hl, Dl = T["linear_attn_config"]["num_heads"], T["linear_attn_config"]["head_dim"]
conv_k = T["linear_attn_config"]["short_conv_kernel_size"]
kvr, rope = T["kv_lora_rank"], T["qk_rope_head_dim"]
topk, kpool = T["index_topk"], T["index_kpool"]

# 状态占用的两处 dtype/宽度依据（不可凭印象）：
#   循环状态：modeling_glm5_next.py 中 update_recurrent_state(last_recurrent_state.to(torch.float32))
#             → fp32，形状 [num_heads, head_dim, head_dim]
#   conv 状态：cache_utils.py lazy_initialization 按 conv_kernel_size 分配
#             torch.zeros((*conv_states.shape[:-1], conv_kernel_size), dtype=conv_states.dtype)
#             → 宽度等于 kernel（4，不是 kernel-1），dtype 随激活（bf16）
REC_ELEMS = Hl * Dl * Dl
REC_BYTES = 4                      # fp32
CONV_ELEMS = 3 * Hl * Dl * conv_k  # q/k/v 三路拼接，宽度 = kernel
CONV_BYTES = 2                     # bf16，随激活 dtype

# DSA 层每 token 需要缓存两部分，都是解码必需状态：
#   ① MLA 潜向量：kv_lora_rank + qk_rope_head_dim
#   ② indexer 的 packed_states：源码 cat([k, gate_scores, valid_channel])
#      = index_head_dim + index_head_dim + 1，经 update_indexer 逐 token 累积
#      （get_pooled_states 每步都要从该缓存重建候选池，不能不存）
IDX_ELEMS = T["index_head_dim"] * 2 + 1
MLA_ELEMS = kvr + rope
DSA_ELEMS = MLA_ELEMS + IDX_ELEMS
KV_BYTES = 2                       # bf16

banner("探针 8：混合架构在长上下文下的实际收益")
print(f"      层数 {L} = KDA {n_kda} + DSA {n_dsa}（比例 {n_kda}:{n_dsa} ≈ 3:1，4 层一循环）")
print(f"      KDA 每层每序列：循环状态 {Hl}*{Dl}*{Dl} = {REC_ELEMS:,} 元素 @fp32 "
      f"= {REC_ELEMS*REC_BYTES/2**20:.2f} MiB")
print(f"                      conv 状态 {3*Hl*Dl}*{conv_k} = {CONV_ELEMS:,} 元素 @bf16 "
      f"= {CONV_ELEMS*CONV_BYTES/2**20:.2f} MiB")
print(f"      DSA 每层每 token：MLA 潜向量 {kvr}+{rope} = {MLA_ELEMS}")
print(f"                        indexer packed {T['index_head_dim']}*2+1 = {IDX_ELEMS}")
print(f"                        合计 {DSA_ELEMS} 元素 @bf16 = {DSA_ELEMS*KV_BYTES/1024:.3f} KiB")

# ---------- 8.1 逐长度对照 ----------
banner("8.1 KV/状态占用随上下文长度变化")
kda_const = n_kda * (REC_ELEMS * REC_BYTES + CONV_ELEMS * CONV_BYTES)
print(f"      KDA 侧常数开销 = {kda_const/2**20:.2f} MiB（与长度无关）")
print()
print(f"      {'上下文':>9s} {'真实混合':>12s} {'假想全DSA':>12s} {'节省':>8s} "
      f"{'KDA占比':>8s}")
for S in [4096, 32768, 131072, 262144, 1048576]:
    dsa_kv = n_dsa * S * DSA_ELEMS * KV_BYTES
    real = kda_const + dsa_kv
    all_dsa = L * S * DSA_ELEMS * KV_BYTES
    print(f"      {S:>9,} {real/2**30:>10.3f}GiB {all_dsa/2**30:>10.3f}GiB "
          f"{(1-real/all_dsa)*100:>7.2f}% {kda_const/real*100:>7.2f}%")
print(f"      注：长度越大，KDA 的常数开销占比越低，节省比例趋近 "
      f"{(1-n_dsa/L)*100:.1f}%（= 1 - {n_dsa}/{L}）")
print(f"      max_position_embeddings = {T['max_position_embeddings']:,}（1M）")
S_break = kda_const / ((L - n_dsa) * DSA_ELEMS * KV_BYTES)
print(f"      交叉点：S = kda_const / ((45-11)*{DSA_ELEMS}*2) = {S_break:.0f} token")
print(f"        S 小于该值时混合架构反而更占内存")
print(f"      对照：若只算 MLA 潜向量（漏掉 indexer 缓存），交叉点会被高估为 "
      f"{kda_const/((L-n_dsa)*MLA_ELEMS*KV_BYTES):.0f} token")

# ---------- 8.2 DSA 打分与命中 ----------
banner("8.2 DSA indexer 的打分量与稀疏命中（k-pool 压缩）")
print(f"      index_topk={topk}  index_kpool={kpool}  select_k = topk//kpool = {topk//kpool} 池")
print(f"      always_select_tail={T['index_kpool_always_select_tail']} → 输出宽度 {topk+kpool-1}")
print()
print(f"      {'S':>9s} {'候选池数':>9s} {'实选池':>7s} {'覆盖token':>10s} {'注意力稀疏度':>12s} "
      f"{'打分量 vs 稠密':>15s}")
for S in [1024, 2048, 4096, 16384, 131072, 1048576]:
    npool = -(-S // kpool)
    sel = min(topk // kpool, npool)
    cov = min(sel * kpool, S)
    print(f"      {S:>9,} {npool:>9,} {sel:>7,} {cov:>10,} {cov/S*100:>11.3f}% "
          f"{npool/S*100:>14.1f}%")
print(f"      → 两层节省叠加：① 打分只在池级别做，量降到 1/{kpool}；")
print(f"        ② 真正参与 softmax 注意力的 key 有 {topk} 的硬上限（+尾巴），")
print(f"        S=1M 时注意力计算量相当于 {topk}/1048576 = {topk/1048576*100:.3f}% 的稠密注意力")

# ---------- 8.3 三种配置对比 ----------
banner("8.3 三种架构配置对比（S=131072，1 个序列）")
S = 131072
kda_all = L * (REC_ELEMS * REC_BYTES + CONV_ELEMS * CONV_BYTES)
rows = [
    ("45 层全 DSA", L * S * DSA_ELEMS * KV_BYTES, 0, "长上下文 KV 全部线性增长"),
    ("45 层全 KDA", 0, kda_all, "KV 恒定，但无精确长程检索能力"),
    (f"真实：KDA {n_kda} + DSA {n_dsa}", n_dsa * S * DSA_ELEMS * KV_BYTES, kda_const,
     "3/4 层常数状态 + 1/4 层稀疏精确检索"),
]
print(f"      {'配置':>22s} {'线性部分':>12s} {'常数部分':>12s} {'合计':>11s}")
for name, lin, const, note in rows:
    print(f"      {name:>22s} {lin/2**30:>10.3f}GiB {const/2**30:>10.3f}GiB "
          f"{(lin+const)/2**30:>9.3f}GiB")
    print(f"      {'':>22s} └ {note}")

# ---------- 8.4 位置编码 ----------
banner("8.4 位置信息来源（全模型无 RoPE）")
print(f"      qk_rope_head_dim = {T['qk_rope_head_dim']}  mla_use_nope = {T['mla_use_nope']}")
print(f"      configuration_glm5_next.py validate_architecture()：qk_rope_head_dim > 0 直接 raise")
print(f"      Glm5NextTextModel.forward L1486：position_embeddings=None 传给每一层")
print(f"      → 45 层文本主干没有任何位置编码运算。位置信息的两个来源：")
print(f"        ① KDA 层的因果递推：状态按 g 逐 token 衰减，天然带顺序")
print(f"        ② KDA 层的 depthwise conv1d（kernel={conv_k}）：提供局部相对位置")
print(f"        ③ DSA 层本身位置无关，依赖上游 KDA 层已注入的位置信息")
print(f"      视觉塔另有独立 RoPE（Glm5NextVisionRotaryEmbedding，theta=10000，2D h/w 位置）")
print(f"      推论（本页标注为推断）：KDA 层承担位置编码职责，是 34:11 这个比例"
      f"以及 layer 0 就是 KDA 的结构约束来源之一")
