# DSA prefill/decode 等价性的完整证据链（最终版，输出存档 diag_final.out）。
#
# 背景结论（diag_layers/diag_layer0/diag_ties 已定位）：
#   prefill(48 token 整段) vs prefill(47)+decode(1) 的最终 logits 在 fp32/fp64 下均有 ~1.6e-3
#   量级差异；根因假说是 DSA indexer 的 ReLU 精确 0 分数在 top-k 边界并列，
#   torch.topk 的并列选择依赖张量长度（47 vs 48 列），选中键集合不同 -> K/V 不同 -> 输出不同。
#
# 本脚本固化五个论断：
#   [1] indexer 分数按官方代码在 fp32 中计算（modeling L250/L256 的 .float() 强转，
#       与模型 dtype 无关）。不同候选宽度的 fp32 matmul 有少量末位舍入差异（量化：
#       仅因果下三角区域，差异条目数 / max_abs / 相对量级），远小于非零分数间隔。
#   [2] 同一个分数张量、仅候选列数不同（47 列 vs 48 列切片）时 top-16 集合发散
#       —— 直接证明 torch.topk 并列选择依赖张量宽度，与数值差异无关。
#   [3] topk 集合发散只出现在「差集位置分数精确为 0.0」的 query 上，且两侧（整段/增量）皆为精确 0.0。
#   [4] 给分数加每列 +1e-9 扰动破坏并列后，top-16 集合完全一致（选择规则本身等价）。
#   [5] index_topk >= 序列长（稀疏化退化为全选）时，整段 vs 增量 logits 一致（fp32, ~3.6e-7）。
#
# 对应源码：modeling_hy_v4.py L250-269（ReLU 打分 + topk）；L268: topk = min(index_topk, T)。
import torch
import torch.nn.functional as F
from mini_model import small_config, HYV4ForCausalLM, HYV4Indexer, apply_rotary_pos_emb

torch.manual_seed(0)
model = HYV4ForCausalLM(small_config(index_topk=16)).double()
model.eval()

cap = {}


def layer_hook(li):
    def hook(m, args, output):
        cap.setdefault(li, []).append(output[1].detach().clone())
    return hook


for li, layer in enumerate(model.model.layers):
    layer.register_forward_hook(layer_hook(li))

# ---- 复刻官方 indexer forward 逐行，仅追加记录 index_scores ----
scores_cap = {}
orig_forward = HYV4Indexer.forward


def patched(self, hidden_states, q_resid, position_embeddings, attention_mask, position_ids, past_key_values=None):
    batch_size, seq_len, _ = hidden_states.shape
    cos, sin = position_embeddings
    q = self.wq_b(q_resid).view(batch_size, seq_len, self.n_heads, self.head_dim)
    q_pass, q_rot = torch.split(q, [self.head_dim - self.qk_rope_head_dim, self.qk_rope_head_dim], dim=-1)
    k = self.k_norm(self.wk(hidden_states).to(self.k_norm.weight.dtype)).to(hidden_states.dtype).unsqueeze(2)
    k_pass, k_rot = torch.split(k, [self.head_dim - self.qk_rope_head_dim, self.qk_rope_head_dim], dim=-1)
    q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
    q = torch.cat([q_pass, q_rot], dim=-1)
    k = torch.cat([k_pass, k_rot], dim=-1).squeeze(2)
    if past_key_values is not None:
        k = past_key_values.update_indexer(k, self.layer_idx)
    scores = F.relu(torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1)))
    weights = (
        self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float()
        * (self.n_heads ** -0.5) * self.softmax_scale
    )
    index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)
    if attention_mask.dtype == torch.bool:
        index_scores = index_scores.masked_fill(~attention_mask, float("-inf"))
    else:
        index_scores = index_scores + attention_mask
    scores_cap.setdefault(self.layer_idx, []).append(index_scores.detach().clone())
    topk = min(self.index_topk, index_scores.shape[-1])
    return index_scores.topk(topk, dim=-1).indices.to(torch.int32)


HYV4Indexer.forward = patched
try:
    seq = torch.randint(0, 1000, (1, 48))
    with torch.no_grad():
        cap.clear(); scores_cap.clear()
        model(seq, use_cache=False)
        FT = {li: v[0] for li, v in cap.items()}
        FS = {li: v[0] for li, v in scores_cap.items()}

        cap.clear(); scores_cap.clear()
        model(seq[:, :-1], use_cache=True)
        PT = {li: v[0] for li, v in cap.items()}
        PS = {li: v[0] for li, v in scores_cap.items()}

    li = 0
    a = FS[li][0, :-1, :47]   # 整段 48-token 前向：query 0..46 x key 0..46（fp64）
    b = PS[li][0]             # 增量前向第一段(47 token)：query 0..46 x key 0..46（fp64）

    # [1] 分数差异量化：只比较因果下三角（query t 可见 key 0..t），掩码位置（finfo.min）除外
    causal = torch.tril(torch.ones(47, 47, dtype=torch.bool))
    a_c, b_c = a[causal], b[causal]
    neq = (a_c != b_c).sum().item()
    max_abs = (a_c - b_c).abs().max().item()
    scale = a_c.abs().max().item()
    print(f"[1] layer0 indexer scores computed in fp32 (.float() in official code):")
    print(f"    causal entries compared={a_c.numel()}, unequal={neq}, max_abs_diff={max_abs:.3e}, "
          f"unmasked score scale={scale:.3e}, rel={max_abs / scale:.3e}")
    # 非零分数对之间也存在小于舍入差异的间隔（2.9e-11 < 1.2e-9），即舍入差异理论上
    # 也能扰动非零分数的排序；但实测发散位置全部是精确 0.0（论断 3），且宽度单独
    # 即可复现发散（论断 2），故发散由 ReLU-0 并列 + 宽度依赖的并列选择主导。
    print(f"    (note: min gap between distinct nonzero scores can be smaller than rounding; "
          f"observed divergence is nonetheless exclusively at exact-0.0 scores, see [3]/[2])")

    # [2] 同一张量、仅候选宽度不同：47 列 vs 48 列切片的 top-16 集合对比
    a_full = FS[li][0]  # [48, 48] 整段分数
    n_div_width = 0
    for t in range(47):
        s47 = set(a_full[t, :47].topk(16).indices.tolist())
        s48 = set(a_full[t, :48].topk(16).indices.tolist())
        if s47 != s48:
            n_div_width += 1
    print(f"[2] same-score-tensor width test (47 cols vs 48 cols): diverging queries = {n_div_width}/47")

    # [3] 实际发散位置：两侧分数是否都精确为 0.0
    ta, tb = FT[li][0, :-1], PT[li][0]
    n_div, n_diffpos, n_zero_a, n_zero_b = 0, 0, 0, 0
    for t in range(47):
        sa, sb = set(ta[t].tolist()), set(tb[t].tolist())
        if sa == sb:
            continue
        n_div += 1
        for j in (sa - sb) | (sb - sa):
            n_diffpos += 1
            if a[t, j].item() == 0.0:
                n_zero_a += 1
            if b[t, j].item() == 0.0:
                n_zero_b += 1
    print(f"[3] layer0 diverging queries: {n_div}/47; diff positions: {n_diffpos}; "
          f"score==0.0 exactly: full-run {n_zero_a}/{n_diffpos}, incremental-run {n_zero_b}/{n_diffpos}")

    # [4] 每列 +1e-9 扰动破坏并列后 top-16 集合完全一致
    pert_a = a + 1e-9 * torch.arange(47, dtype=torch.float64)
    pert_b = b + 1e-9 * torch.arange(47, dtype=torch.float64)
    pa, pb = pert_a.topk(16, dim=-1).indices, pert_b.topk(16, dim=-1).indices
    set_eq = all(set(pa[t].tolist()) == set(pb[t].tolist()) for t in range(47))
    print(f"[4] tie-broken top-16 set-equal (per-column +1e-9 perturbation): {set_eq}")
finally:
    HYV4Indexer.forward = orig_forward

# [5] index_topk >= 序列长：稀疏化退化为全选，整段 vs 增量 logits 一致
torch.manual_seed(0)
m2 = HYV4ForCausalLM(small_config(index_topk=1024))
m2.eval()
seq2 = torch.randint(0, 1000, (1, 48))
with torch.no_grad():
    full = m2(seq2, use_cache=False).logits
    inc = m2(seq2[:, :-1], use_cache=True)
    last = m2(seq2[:, -1:], past_key_values=inc.past_key_values).logits
print(f"[5] no-sparsity (index_topk=1024 >= T=48) logits max_abs diff (fp32): "
      f"{(full[:, -1] - last[:, -1]).abs().max().item():.3e}")
