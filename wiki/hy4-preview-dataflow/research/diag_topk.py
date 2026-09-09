# 诊断：prefill 与 decode 的 logits 差 2.8e-2 的根因。
# 假设：indexer ReLU 后大量并列 0 分数，torch.topk 在不同尺寸张量上并列选择不同，
# 导致同一 query 在整段/增量前向中选出不同的 top-k 集合。
# 方法：hook DecoderLayer 捕获 topk_indices，两条路径逐层对比。
import torch
from mini_model import small_config, HYV4ForCausalLM, HYV4DecoderLayer

torch.manual_seed(0)
cfg = small_config()
model = HYV4ForCausalLM(cfg)
model.eval()

captured = []


def mk_hook(li):
    def hook(mod, args, output):
        # DecoderLayer.forward 返回 (hidden_states, topk_indices)
        _, topk = output
        captured.append((li, topk))
    return hook


for li, layer in enumerate(model.model.layers):
    layer.register_forward_hook(mk_hook(li))


def run(seq, incremental):
    captured.clear()
    with torch.no_grad():
        if not incremental:
            full = model(seq, use_cache=False).logits
            return full, {li: t for li, t in captured}
        o = model(seq[:, :1], use_cache=True)
        for t in range(1, seq.shape[1]):
            o = model(seq[:, t : t + 1], past_key_values=o.past_key_values)
        return o.logits, {li: torch.cat([c[1] for c in captured if c[0] == li], dim=1)
                          for li in range(len(model.model.layers))}


seq = torch.randint(0, 1000, (1, 40))
full_logits, full_topk = run(seq, False)
inc_logits, inc_topk = run(seq, True)

print(f"logits max_abs diff: {(full_logits - inc_logits).abs().max().item():.3e}")
for li in range(len(model.model.layers)):
    a, b = full_topk[li], inc_topk[li]
    same = (a == b).all().item()
    # 集合级别对比（每 query 的 topk 集合，与顺序无关）
    B, S, K = a.shape
    set_same = all(
        set(a[i, s].tolist()) == set(b[i, s].tolist())
        for i in range(B) for s in range(S)
    )
    print(f"layer {li:2d}: topk identical={same}  set-equal={set_same}  shape={tuple(a.shape)}")
