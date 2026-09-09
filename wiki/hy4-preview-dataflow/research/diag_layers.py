# 诊断 2：逐层比较整段前向与增量前向的 hidden_states，定位首个发散层。
import torch
from mini_model import small_config, HYV4ForCausalLM

torch.manual_seed(0)
cfg = small_config()
model = HYV4ForCausalLM(cfg)
model.eval()

cap = {}


def mk_hook(li):
    def hook(mod, args, output):
        cap.setdefault(li, []).append(output[0].detach().clone())
    return hook


for li, layer in enumerate(model.model.layers):
    layer.register_forward_hook(mk_hook(li))

seq = torch.randint(0, 1000, (1, 40))
with torch.no_grad():
    cap.clear()
    full = model(seq, use_cache=False).logits
    full_h = {li: cap[li][0] for li in cap}  # [1,40,4,512]

    cap.clear()
    o = model(seq[:, :1], use_cache=True)
    for t in range(1, 40):
        o = model(seq[:, t : t + 1], past_key_values=o.past_key_values)
    inc_h = {li: torch.cat(cap[li], dim=1) for li in cap}  # [1,40,4,512]

print(f"final logits diff: {(full - o.logits).abs().max().item():.3e}")
for li in range(10):
    d = (full_h[li] - inc_h[li]).abs().max().item()
    print(f"layer {li} out max_abs_diff: {d:.3e}")
# 每层内部的分歧位置分布
li_first = None
for li in range(10):
    d = (full_h[li] - inc_h[li]).abs().max().item()
    if d > 1e-4 and li_first is None:
        li_first = li
        per_pos = (full_h[li] - inc_h[li]).abs().amax(dim=(0, 2, 3))
        bad = (per_pos > 1e-4).nonzero().flatten().tolist()
        print(f"first diverging layer {li}: diverging positions = {bad}")
