# 诊断 3：整段 48 vs 段 47 前向，layer 0 内部逐步对比，找 1e-3 差异的放大点。
# 同时直接捕获官方 forward 的 topk_indices（不经重算）。
import torch
from mini_model import small_config, HYV4ForCausalLM

torch.manual_seed(0)
cfg = small_config()
model = HYV4ForCausalLM(cfg)
model.eval()

cap = {"sub": {}, "topk": {}}


def sub_hook(name):
    def hook(m, args, out):
        if isinstance(out, tuple):
            out = out[0]
        cap["sub"].setdefault(name, []).append(out.detach().clone())
    return hook


def layer_hook(li):
    def hook(m, args, output):
        cap["topk"].setdefault(li, []).append(output[1].detach().clone())
    return hook


L0 = model.model.layers[0]
L0.attn_hc.register_forward_hook(sub_hook("attn_hc"))
L0.input_layernorm.register_forward_hook(sub_hook("input_ln"))
L0.self_attn.register_forward_hook(sub_hook("attn_out"))
L0.ffn_hc.register_forward_hook(sub_hook("ffn_hc"))
L0.post_attention_layernorm.register_forward_hook(sub_hook("post_ln"))
L0.mlp.register_forward_hook(sub_hook("mlp_out"))
for li, layer in enumerate(model.model.layers):
    layer.register_forward_hook(layer_hook(li))

seq = torch.randint(0, 1000, (1, 48))
with torch.no_grad():
    cap["sub"].clear(); cap["topk"].clear()
    model(seq, use_cache=False)
    full = {k: v[0] for k, v in cap["sub"].items()}
    full_topk = {li: v[0] for li, v in cap["topk"].items()}

    cap["sub"].clear(); cap["topk"].clear()
    model(seq[:, :-1], use_cache=True)
    pre = {k: v[0] for k, v in cap["sub"].items()}
    pre_topk = {li: v[0] for li, v in cap["topk"].items()}

print("layer-0 internal (first 47 positions), full[48] vs pre[47]:")
for k in ["attn_hc", "input_ln", "attn_out", "ffn_hc", "post_ln", "mlp_out"]:
    a, b = full[k][:, :47], pre[k][:, :47]
    print(f"  {k:10s}: max_abs={(a - b).abs().max().item():.3e}  shape={tuple(a.shape)}")

# 官方 topk 对比（前 47 query）
for li in [0, 1, 5, 9]:
    a, b = full_topk[li][:47], pre_topk[li]
    same = (a == b).all().item()
    set_eq = all(set(a[t].tolist()) == set(b[t].tolist()) for t in range(47))
    print(f"official topk layer {li}: identical={same} set-equal={set_eq}")

# attention 输入也对比：hc 输出的 out（self_attn 的真正输入 = input_layernorm(attn_hc.out)）
# attn_hc 返回 tuple(post, out) —— hook 里取了 out[0]=post！修正：直接在下面重抓
print("\n[fix] attn_hc returns (post, out); hook captured post. Re-check with both:")
cap2 = {}


def hc_hook(m, args, out):
    cap2.setdefault("post", []).append(out[0].detach().clone())
    cap2.setdefault("out", []).append(out[1].detach().clone())


L0.attn_hc.register_forward_hook(hc_hook)
with torch.no_grad():
    cap2.clear()
    model(seq, use_cache=False)
    f_post, f_out = cap2["post"][0], cap2["out"][0]
    cap2.clear()
    model(seq[:, :-1], use_cache=True)
    p_post, p_out = cap2["post"][0], cap2["out"][0]
print(f"  attn_hc post: max_abs={(f_post[:, :47] - p_post).abs().max().item():.3e}")
print(f"  attn_hc out : max_abs={(f_out[:, :47] - p_out).abs().max().item():.3e}")
