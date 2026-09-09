# 验证目标：长 prefill（T=47）与整段（T=48）在 layer 0 的 indexer 输出差异根因。
# 假设：indexer 分数逐位一致（机制等价），差异全部来自 ReLU 并列 0 分数下
# torch.topk 的并列选择随张量长度（47 vs 48 列）变化。
# 判据：
#   (a) 两条路径捕获的 index_scores 对前 47 个 query 逐位一致；
#   (b) topk_indices 的差集位置在两条路径下的分数都恰好为 0（并列）；
#   (c) 每个发散 query 的可见 ReLU-0 分数个数 > topk（并列超饱和）。
import torch
import torch.nn.functional as F
from mini_model import small_config, HYV4ForCausalLM

torch.manual_seed(0)
cfg = small_config()
model = HYV4ForCausalLM(cfg)
model.eval()

scores_cap = {}


def score_hook(li):
    def hook(mod, args, output):
        scores_cap.setdefault(li, []).append((args, output))
    return hook


# HYV4Indexer.forward 返回 topk indices；分数要自己重算 —— 直接 hook indexer 内部：
# 更简单：用 forward_pre_hook 抓输入，再手动调用 indexer 的中间步骤复现分数。
# 这里直接 monkey-patch HYV4Indexer.forward 太侵入；改为在 indexer 模块上加 wrapper。
from mini_model import HYV4Indexer

orig_forward = HYV4Indexer.forward


def patched(self, hidden_states, q_resid, position_embeddings, attention_mask, position_ids, past_key_values=None):
    # 逐行复刻官方 forward（源码 198-269 行），额外记录 softmax 前的 index_scores
    batch_size, seq_len, _ = hidden_states.shape
    cos, sin = position_embeddings
    q = self.wq_b(q_resid)
    q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
    q_pass, q_rot = torch.split(q, [self.head_dim - self.qk_rope_head_dim, self.qk_rope_head_dim], dim=-1)
    k = self.k_norm(self.wk(hidden_states).to(self.k_norm.weight.dtype)).to(hidden_states.dtype).unsqueeze(2)
    k_pass, k_rot = torch.split(k, [self.head_dim - self.qk_rope_head_dim, self.qk_rope_head_dim], dim=-1)
    from mini_model import apply_rotary_pos_emb
    q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
    q = torch.cat([q_pass, q_rot], dim=-1)
    k = torch.cat([k_pass, k_rot], dim=-1).squeeze(2)
    if past_key_values is not None:
        k = past_key_values.update_indexer(k, self.layer_idx)
    scores = torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1))
    scores = F.relu(scores)
    weights = (
        self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float()
        * (self.n_heads ** -0.5)
        * self.softmax_scale
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
        scores_cap.clear()
        full = model(seq, use_cache=False).logits
        full_scores = {li: s[0] for li, s in scores_cap.items()}

        scores_cap.clear()
        inc = model(seq[:, :-1], use_cache=True)
        pre_scores = {li: s[0] for li, s in scores_cap.items()}

        scores_cap.clear()
        last = model(seq[:, -1:], past_key_values=inc.past_key_values).logits

    li = 0
    a, b = full_scores[li][0, :-1, :47], pre_scores[li][0]  # [47,47] vs [47,47]（前 47 列）
    # (a) 前 47 列分数逐位一致
    print(f"(a) scores first-47-cols bitwise equal: {(a == b).all().item()}")
    n_mismatch = (a != b).sum().item()
    print(f"    mismatched entries: {n_mismatch}")

    # (b) topk 差集的分数是否全为 0（并列）
    ka = a.topk(16, dim=-1).indices  # [47,16] from 48 cols
    kb = b.topk(16, dim=-1).indices  # [47,16] from 47 cols
    n_div, n_zero_in_diff, n_diff = 0, 0, 0
    for t in range(47):
        sa, sb = set(ka[t].tolist()), set(kb[t].tolist())
        if sa == sb:
            continue
        n_div += 1
        for j in (sa - sb) | (sb - sa):
            n_diff += 1
            if a[t, j].item() == 0.0 and b[t, j].item() == 0.0:
                n_zero_in_diff += 1
    print(f"(b) diverging queries: {n_div}/47, diff positions: {n_diff}, of which score==0 in both: {n_zero_in_diff}")

    # (c) 每个 query 的可见 0 分数个数（并列饱和度）
    zeros_visible = []
    for t in range(47):
        vis = a[t, : t + 1]  # 因果可见
        zeros_visible.append(int((vis == 0.0).sum()))
    import statistics
    print(f"(c) visible ReLU-0 count per query: min={min(zeros_visible)} median={statistics.median(zeros_visible)} "
          f"max={max(zeros_visible)} (topk=16)")
finally:
    HYV4Indexer.forward = orig_forward
