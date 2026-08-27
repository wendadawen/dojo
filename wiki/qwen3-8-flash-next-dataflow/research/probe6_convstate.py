"""排查 5E 差异根因：GDN 的 causal conv1d 状态在逐 token decode 时必须跨步保留。

假设：probe5 的缩小实现里 conv1d 用 padding=3 在每个 decode 步独立卷积，
      等于每步都把前 3 个 token 当成 0，因此与 prefill 不一致。
      官方源码用 cache_params.update_conv_state 维护一个长度 kernel-1 的滑窗。

对应源码：transformers@36deb0b5 modeling_qwen4_exp.py
  L474-483  decode 单步走 causal_conv1d_update，就地更新 conv_state
  L485-496  prefill 走 causal_conv1d_fn，先 update_conv_state 再卷积
  L500      mixed_qkv = mixed_qkv[:, :, -seq_len:]  丢掉前置状态对应的输出
"""
import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(0)
CD, KS = 12, 4          # conv_dim, kernel_size
conv = nn.Conv1d(CD, CD, KS, groups=CD, padding=KS - 1, bias=False)
T = 10
x = torch.randn(1, T, CD)

# ---- prefill：整段卷积 ----
full = F.silu(conv(x.transpose(1, 2))[..., :T]).transpose(1, 2)

print("=" * 72)
print("A. 错误做法：decode 每步独立卷积（等于把历史当 0）")
print("=" * 72)
bad = []
for t in range(T):
    step = x[:, t:t + 1]
    o = F.silu(conv(step.transpose(1, 2))[..., :1]).transpose(1, 2)
    bad.append(o)
bad = torch.cat(bad, dim=1)
print(f"  与 prefill 最大绝对差 = {(full - bad).abs().max().item():.4e}")
print(f"  逐位置差: {[round((full[0,t]-bad[0,t]).abs().max().item(),4) for t in range(T)]}")
print(f"  -> 每个位置都错，因为丢了 kernel-1={KS-1} 个历史 token")

print()
print("=" * 72)
print("B. 正确做法：维护长度 kernel-1 的滑窗状态（对应官方 update_conv_state）")
print("=" * 72)
state = torch.zeros(1, CD, KS - 1)      # conv_state
good = []
for t in range(T):
    step = x[:, t:t + 1].transpose(1, 2)              # [1, CD, 1]
    win = torch.cat([state, step], dim=-1)            # [1, CD, KS]
    o = F.silu(conv(win)[..., KS - 1:KS])             # 取对应当前 token 的那一个输出
    good.append(o.transpose(1, 2))
    state = win[..., 1:]                              # 滑窗前移
good = torch.cat(good, dim=1)
d = (full - good).abs().max().item()
print(f"  与 prefill 最大绝对差 = {d:.4e}")
print(f"  逐位置差: {[f'{(full[0,t]-good[0,t]).abs().max().item():.1e}' for t in range(T)]}")
print(f"  判定（< 1e-5）: {'一致' if d < 1e-5 else '不一致'}")
print()
print("结论：GDN 的 depthwise causal conv 必须跨 decode 步保留 kernel-1 个通道状态。")
print(f"      真实配置 linear_conv_kernel_dim=4 -> 每层需保留 3 个位置 x conv_dim=10240 通道。")
print("      probe5 的缩小实现未维护该状态，这是 5E 出现 4.4e-01 差异的原因，")
print("      与 GDN 递归本身无关（递归等价性已在 probe1 实测 1A 得到 1.36e-07 的一致性）。")
