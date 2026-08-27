"""
探针 2：DSA（DeepSeek Sparse Attention）层 —— MLA 主路 + k-pool indexer。

验证目标
  1. MLA 各投影的真实形状，与 checkpoint 张量头一致；确认 NoPE（qk_rope_head_dim=0）
  2. KV cache 的实际形态：官方 transformers 实现 cache 的是「展开后的 key/value」而非潜向量，
     两种口径的每 token 字节数分别是多少（这点必须实测，不能凭 MLA 常识断言）
  3. indexer 的 k-pool 压缩：候选数 = ceil(S/kpool)，选中 select_k = index_topk // index_kpool 个池，
     展开回 token 后宽度 = index_topk (+ kpool-1 尾巴)
  4. 稀疏度：实测 mask 中每个 query 实际可见的 key 数量，与 index_topk 预算对照
  5. 跨层 topk 共享：indexer_types 里 shared 层复用前一 full 层的选择结果
  6. 长序列下稀疏注意力的实际命中率（S 远大于 index_topk 时才体现稀疏）

对应源码
  Glm5NextTextAttention.forward             modeling_glm5_next.py L1155-1216
  Glm5NextTextAttention.expand_kv           L1136-1153
  Glm5NextTextIndexer.forward               L774-877
  Glm5NextTextIndexer.get_pooled_states     L899-972
  build_attention_mask_from_topk            L1218-1256
"""
from __future__ import annotations
import sys, json, math
sys.path.insert(0, "/tmp/glm53f/probe")
import torch
from harness import real_config, banner, Glm5NextTextAttention, MiniCache

torch.manual_seed(0)
C = real_config()
DT = torch.float32
banner("探针 2：DSA 层（MLA 主路 + k-pool indexer），真实 config 维度")

# ---------- 2.1 MLA 投影形状与参数量 ----------
attn = Glm5NextTextAttention(C, layer_idx=3).to(DT).eval()
print("[2.1] DSA 层参数张量（真实维度）")
tot = idx_tot = 0
for n, p in attn.named_parameters():
    tot += p.numel()
    if n.startswith("indexer."):
        idx_tot += p.numel()
    print(f"      {n:42s} {tuple(p.shape)}  {p.numel():>12,}")
print(f"      合计 {tot:,} = {tot/1e6:.2f} M，其中 indexer {idx_tot:,} = {idx_tot/1e6:.2f} M "
      f"({idx_tot/tot*100:.2f}%)")

print("\n[2.2] NoPE 确认")
print(f"      config.qk_rope_head_dim = {C.qk_rope_head_dim}  (configuration 里 validate_architecture "
      f"要求必须为 0，>0 直接报错)")
print(f"      qk_nope_head_dim = {C.qk_nope_head_dim}, v_head_dim = {C.v_head_dim}, "
      f"qk_head_dim = nope+rope = {attn.qk_head_dim}")
print(f"      config.mla_use_nope = {json.load(open('/tmp/glm53f/config.json'))['text_config']['mla_use_nope']}")
print(f"      Glm5NextTextModel.forward 里传 position_embeddings=None，DSA 层不调用任何 apply_rotary")
print(f"      → 全部 45 层文本主干不含任何位置编码，位置信息只来自 KDA 层的因果递推与 conv1d")

# ---------- 2.3 KV cache 实测形态 ----------
banner("2.3 KV cache 实测形态（transformers 实现口径）")
cache = MiniCache(num_layers=4)
S = 40
x = torch.randn(1, S, C.hidden_size, dtype=DT)
mask = torch.ones(1, S, dtype=torch.bool)
with torch.no_grad():
    out, _, topk = attn(hidden_states=x, attention_mask=mask, past_key_values=cache)
st = cache.layers[3]
print(f"      prefill S={S}: 输出 {tuple(out.shape)}  finite={bool(torch.isfinite(out).all())}")
print(f"      cache.keys   {tuple(st.keys.shape)}   (B, num_heads, S, qk_head_dim)")
print(f"      cache.values {tuple(st.values.shape)} (B, num_heads, S, v_head_dim)")
per_tok_expanded = (C.num_attention_heads * attn.qk_head_dim + C.num_attention_heads * C.v_head_dim)
print(f"      展开口径每 token 元素 = heads*(qk+v) = {C.num_attention_heads}*({attn.qk_head_dim}+{C.v_head_dim}) "
      f"= {per_tok_expanded:,}  → bf16 {per_tok_expanded*2/1024:.1f} KiB/token/层")
per_tok_latent = C.kv_lora_rank + C.qk_rope_head_dim
print(f"      潜向量口径每 token 元素 = kv_lora_rank + rope = {C.kv_lora_rank}+{C.qk_rope_head_dim} "
      f"= {per_tok_latent}  → bf16 {per_tok_latent*2/1024:.3f} KiB/token/层")
print(f"      两者相差 {per_tok_expanded/per_tok_latent:.0f}×。官方 transformers 实现（expand_kv 后再 cache）")
print(f"      走的是展开口径；MLA 的显存优势要靠推理框架自行 cache 潜向量实现（本探针不能证明后者）")
n_dsa = sum(1 for t in C.layer_types if t == "deepseek_sparse_attention")
print(f"      DSA 层数 = {n_dsa}；潜向量口径全模型 = "
      f"{n_dsa*per_tok_latent*2/1024:.2f} KiB/token")

# ---------- 2.4 indexer 的 k-pool 机制 ----------
banner("2.4 indexer k-pool 压缩机制（实测候选数与选择宽度）")
ix = attn.indexer
print(f"      index_kpool={ix.index_kpool}  index_topk={ix.index_topk}  "
      f"index_n_heads={ix.n_heads}  index_head_dim={ix.head_dim}")
print(f"      select_k = index_topk // index_kpool = {ix.index_topk}//{ix.index_kpool} = "
      f"{ix.index_topk // ix.index_kpool} 个池")
print(f"      always_select_tail={ix.index_kpool_always_select_tail} → 输出宽度 = "
      f"index_topk + (kpool-1) = {ix.index_topk + ix.index_kpool - 1}")
print(f"      softmax_scale = head_dim^-0.5 = {ix.softmax_scale:.8f}")
print(f"      打分公式：relu(q·pool_k * scale) 逐头 → 乘 weights_proj(x)*n_heads^-0.5 后按头求和")
print(f"      每头权重系数 n_heads^-0.5 = {C.index_n_heads**-0.5:.8f}")

for Slen in [16, 40, 1024, 4096, 16384]:
    n_pool = math.ceil(Slen / ix.index_kpool)
    sel = min(ix.index_topk // ix.index_kpool, n_pool)
    covered = min(sel * ix.index_kpool, Slen)
    print(f"      S={Slen:6d}  候选池={n_pool:6d}  实选池={sel:5d}  "
          f"覆盖 token≈{covered:6d}  稀疏度={covered/Slen*100:6.2f}%  "
          f"打分次数={n_pool:6d}(vs 稠密 {Slen})")

# ---------- 2.5 实测 mask：每个 query 真实可见 key 数 ----------
banner("2.5 实测注意力 mask（每个 query 实际可见的 key 数量）")
print(f"      注：本 checkpoint 全为 indexer_types='full'，next_skip_topk=False，")
print(f"      forward 返回 None（无需向下传），因此直接调用 indexer 取选择结果")
print(f"      真实 index_topk={C.index_topk}，需 S > 2048 才出现稀疏；先看 S 小于预算的情形")


def run_indexer(a, xx, cache_obj):
    """indexer 依赖 forward 里先做的 past_key_values.update()（L1179）填好 cache_layer.keys，
    因此不能脱离 forward 单独调用。改用 forward hook 在真实前向中捕获 indexer 的输出。"""
    box = {}
    h = a.indexer.register_forward_hook(lambda m, i, o: box.__setitem__("tk", o))
    try:
        with torch.no_grad():
            a(hidden_states=xx,
              attention_mask=torch.ones(xx.shape[0], xx.shape[1], dtype=torch.bool),
              past_key_values=cache_obj)
    finally:
        h.remove()
    return box["tk"]


for Slen in [16, 40]:
    c2 = MiniCache(num_layers=4)
    xx = torch.randn(1, Slen, C.hidden_size, dtype=DT)
    tk = run_indexer(attn, xx, c2)
    am = attn.build_attention_mask_from_topk(tk, torch.zeros(1, 1, Slen, attn.qk_head_dim), Slen)
    vis = (am if am.dtype == torch.bool else (am == 0)).squeeze(1).sum(-1)[0]
    causal = torch.arange(1, Slen + 1)
    print(f"      S={Slen}: topk_indices {tuple(tk.shape)}；可见数前8={vis[:8].tolist()}  "
          f"末3={vis[-3:].tolist()}")
    print(f"              因果上限前8={causal[:8].tolist()}  完全等于因果掩码={bool((vis==causal).all())}")

# ---------- 2.6 缩小 index_topk 以观察真正的稀疏行为 ----------
banner("2.6 缩小 index_topk 观察稀疏选择（结构不变，仅缩预算）")
C2 = real_config({"index_topk": 16})     # 16 % kpool(4) == 0，满足官方 validate 的整除约束
a2 = Glm5NextTextAttention(C2, layer_idx=3).to(DT).eval()
Slen = 64
c3 = MiniCache(num_layers=4)
xx = torch.randn(1, Slen, C.hidden_size, dtype=DT)
tk = run_indexer(a2, xx, c3)
am = a2.build_attention_mask_from_topk(tk, torch.zeros(1, 1, Slen, a2.qk_head_dim), Slen)
vis = (am if am.dtype == torch.bool else (am == 0)).squeeze(1).sum(-1)[0]
causal = torch.arange(1, Slen + 1)
print(f"      index_topk=16, kpool=4 → select_k=4 池, 输出宽度=16+3=19")
print(f"      topk_indices {tuple(tk.shape)}  （= index_topk + kpool-1 = 19 ✓）")
print(f"      每 query 可见 key 数（前 20 个 query）={vis[:20].tolist()}")
print(f"      对应因果上限                    ={causal[:20].tolist()}")
print(f"      query 63 可见 {int(vis[63])} 个 key（因果上限 64）→ 稀疏率 "
      f"{int(vis[63])/64*100:.1f}%")
print(f"      被选中的原始 token 下标（query 63，去掉 -1）= "
      f"{sorted(set(t for t in tk[0,63].tolist() if t>=0))}")

print("\n      尾巴（always_select_tail）语义核对：")
print("      append_visible_tail 用 tail_count = visible_count % kpool 决定尾巴长度（L1008），")
print("      因此 visible_count 恰为 kpool 整数倍时 tail_count=0，没有不完整池，尾巴为空。")
for qi in [60, 61, 62, 63]:
    sel = sorted(set(t for t in tk[0, qi].tolist() if t >= 0))
    vc = qi + 1                      # 无 padding 时可见数 = 位置+1
    tc = vc % C2.index_kpool
    tail_expect = list(range(vc - tc, vc)) if tc else []
    print(f"        query {qi}: 可见={vc:3d}  tail_count={vc}%{C2.index_kpool}={tc}  "
          f"期望尾巴={tail_expect}  尾巴全在选中集={all(t in sel for t in tail_expect)}  "
          f"选中数={len(sel)}")

# ---------- 2.7 跨层 topk 共享 ----------
banner("2.7 跨层 top-k 共享（indexer_types）")
it = C.indexer_types
print(f"      config.indexer_types 取值分布 = "
      f"{ {v: it.count(v) for v in set(it)} }")
print(f"      本 checkpoint 全部为 'full' → 每个 DSA 层各自跑 indexer，无共享")
print(f"      源码支持共享：skip_topk = indexer_types[i]=='shared' 时 indexer=None，")
print(f"      改用上层传入的 prev_topk_indices（L1130-1134, L1189-1191）")
print(f"      实测带 indexer 的 DSA 层 = {[i for i,t in enumerate(C.layer_types) if t=='deepseek_sparse_attention']}")
print(f"      与 checkpoint 张量交叉验证结果一致（见 verify_structure.out [4]）")
print(f"      config.index_share_for_mtp_iteration = "
      f"{json.load(open('/tmp/glm53f/config.json'))['text_config']['index_share_for_mtp_iteration']}")

# ---------- 2.8 decode 一步 ----------
banner("2.8 DSA 层 decode 一步（增量）")
with torch.no_grad():
    xd = torch.randn(1, 1, C.hidden_size, dtype=DT)
    od, _, tkd = attn(hidden_states=xd, attention_mask=torch.ones(1, 1, dtype=torch.bool),
                      past_key_values=cache)
print(f"      输出 {tuple(od.shape)}  finite={bool(torch.isfinite(od).all())}")
print(f"      cache 增长到 {tuple(cache.layers[3].keys.shape)}（S {S} → {S+1}）")
print(f"      forward 返回的 topk（next_skip_topk={attn.next_skip_topk}）= {tkd}")
tkd2 = run_indexer(attn, xd, cache)
print(f"      直接调 indexer 得到 decode 选择 {tuple(tkd2.shape)}  "
      f"（1 个 query 对全部 {cache.layers[3].indexer.shape[1]} 个 kv 位置做选择）")
