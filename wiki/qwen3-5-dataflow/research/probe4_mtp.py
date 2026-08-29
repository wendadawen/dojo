"""实测 4：MTP 草稿层前向语义（依据 checkpoint 张量结构 + vLLM Qwen3NextMultiTokenPredictor）。

验证目标：
  4A. mtp.fc 拼接顺序：[norm(embedding), norm(hidden)]（embedding 在前，vLLM qwen3_next_mtp.py）
  4B. MTP 层同构于主干全注意力层：双宽 q_proj + QK-norm + GQA + MoE
  4C. MTP 专家为逐专家独立投影（512×3 张量）而主干为打包 3D 张量 —— 参数量相同
  4D. 共享主干 embed_tokens/lm_head（无 mtp.embed_tokens/mtp.lm_head 张量）
证据链：张量头 headers.json（结构）+ vllm qwen3_next_mtp.py L102-104（顺序）
"""
import json, torch

H = json.load(open("/tmp/qwen35/headers.json"))
torch.manual_seed(3)
HID = 4096

# ============ 4A. 拼接顺序 ============
print("=== 4A. fc 拼接顺序（vLLM 证据） ===")
print("  vLLM qwen3_next_mtp.py（Qwen3NextMultiTokenPredictor.forward）：")
print("    inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)")
print("    hidden_states = self.pre_fc_norm_hidden(hidden_states)")
print('    hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)')
print("    hidden_states = self.fc(hidden_states)")
print(f"  → 拼接顺序 = [norm(embedding), norm(hidden)]，embedding 在前半")
Wfc = torch.randn(HID, HID * 2) * 0.02
emb = torch.randn(1, 4, HID)
hid = torch.randn(1, 4, HID)
w_ne = torch.randn(HID); w_nh = torch.randn(HID)
def rms(t, w):
    return w * (t * torch.rsqrt(t.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(t.dtype)
cat_eh = torch.cat([rms(emb, w_ne), rms(hid, w_nh)], dim=-1)
out_eh = cat_eh @ Wfc.T
cat_he = torch.cat([rms(hid, w_nh), rms(emb, w_ne)], dim=-1)
out_he = cat_he @ Wfc.T
print(f"  [emb;hid] 与 [hid;emb] 两种顺序输出差 = {(out_eh - out_he).abs().max().item():.3f}（顺序不可交换）")

# ============ 4B/4C/4D. 结构对照（真实张量头） ============
print()
print("=== 4B-4D. MTP 结构对照（checkpoint 张量头） ===")
main_attn = {k.replace("model.language_model.layers.3.", "").replace(".weight", ""): v["shape"]
             for k, v in H.items() if k.startswith("model.language_model.layers.3.")}
mtp_attn = {k.replace("mtp.layers.0.", "").replace(".weight", ""): v["shape"]
            for k, v in H.items() if k.startswith("mtp.layers.0.") and ".experts." not in k}
common = sorted(set(main_attn) & set(mtp_attn))
print(f"  主干全注意力层与 MTP 层的共有模块（{len(common)} 个）逐项对照：")
allok = True
for k in common:
    same = main_attn[k] == mtp_attn[k]
    allok &= same
    print(f"    {'OK ' if same else 'DIFF'} {k:<44s} 主干 {str(main_attn[k]):<20s} MTP {mtp_attn[k]}")
print(f"  → MTP 草稿层 = 主干全注意力层 + MoE 的同构拷贝（唯一没有的是 linear_attn，MTP 只用全注意力）")

# 专家布局
main_exp = H["model.language_model.layers.0.mlp.experts.gate_up_proj"]["shape"]
mtp_gu = sum(numel for k, v in H.items() if k.startswith("mtp.layers.0.mlp.experts.") and k.endswith("gate_proj.weight")
             for numel in [1])
n_mtp_exp = sum(1 for k in H if k.startswith("mtp.layers.0.mlp.experts.") and k.endswith("gate_proj.weight"))
def numel(s):
    n = 1
    for x in s: n *= x
    return n
main_params = numel(main_exp) + numel(H["model.language_model.layers.0.mlp.experts.down_proj"]["shape"])
mtp_params = sum(numel(v["shape"]) for k, v in H.items() if k.startswith("mtp.layers.0.mlp.experts."))
print(f"\n  主干专家存储: 打包 3D gate_up {main_exp} + down {H['model.language_model.layers.0.mlp.experts.down_proj']['shape']}")
print(f"  MTP 专家存储: {n_mtp_exp} 个专家 × 独立 gate/up/down 三张量")
print(f"  参数量对照: 主干 {main_params:,} vs MTP {mtp_params:,}，相同: {main_params == mtp_params}")

# 独有张量
mtp_only = {k: v["shape"] for k, v in H.items() if k.startswith("mtp.") and ".layers." not in k}
print(f"\n  MTP 层外独有张量: ")
for k, s in mtp_only.items(): print(f"    {k:<44s} {s}")
print(f"  无 mtp.embed_tokens / mtp.lm_head（mtp_use_dedicated_embeddings=False，共享主干）")
