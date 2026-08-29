"""独立核算 Qwen3.5-397B-A17B 参数量与单 token 激活量（逐张量累加，非 config 推导）。

对照官方宣称：总参数 397B（"3970 亿"）、单 token 激活 17B（A17B）。
激活口径：每 token 实际参与矩阵乘的参数 —— 注意力取该层实际类型（GDN/全注意力），
MoE 取 top-10 路由专家 + 恒激活共享专家 + 路由器，embed 查表不计、lm_head 计入。
"""
import json, re, collections

H = json.load(open("/tmp/qwen35/headers.json"))
C = json.load(open("/tmp/qwen35/config.json"))
TC, VC = C["text_config"], C["vision_config"]

def numel(shape):
    n = 1
    for s in shape: n *= s
    return n

buckets = collections.Counter()
for k, v in H.items():
    n = numel(v["shape"])
    if k.startswith("mtp."):
        buckets["MTP 草稿层"] += n
    elif k.startswith("model.visual."):
        buckets["视觉塔"] += n
    elif k == "model.language_model.embed_tokens.weight":
        buckets["词嵌入"] += n
    elif k == "lm_head.weight":
        buckets["输出头"] += n
    elif ".mlp.experts." in k:
        buckets["MoE 路由专家"] += n
    elif ".mlp.shared_expert." in k or ".mlp.shared_expert_gate." in k:
        buckets["MoE 共享专家"] += n
    elif ".mlp.gate." in k:
        buckets["MoE 路由器"] += n
    elif ".linear_attn." in k:
        buckets["GDN 线性注意力"] += n
    elif ".self_attn." in k:
        buckets["全注意力"] += n
    elif "layernorm" in k or k.endswith(".norm.weight"):
        buckets["归一化层"] += n
    else:
        buckets["其他"] += n

total = sum(buckets.values())
print("=" * 66)
print("参数量分组（逐张量累加，BF16/F32 同计）")
print("=" * 66)
for k, v in buckets.most_common():
    print(f"  {k:<14s} {v:>18,}  ({v/1e9:8.2f} B, {v/total*100:5.2f}%)")
print(f"  {'合计':<14s} {total:>18,}  ({total/1e9:8.2f} B)")
print()

# 激活量：取一层 GDN + 一层全注意力实际算的
kd = TC["linear_num_key_heads"] * TC["linear_key_head_dim"]
vd = TC["linear_num_value_heads"] * TC["linear_value_head_dim"]
gdn_active = (numel([kd*2+vd, TC["hidden_size"]]) + numel([vd, TC["hidden_size"]]) +
              2*numel([TC["linear_num_value_heads"], TC["hidden_size"]]) +
              numel([kd*2+vd, 1, TC["linear_conv_kernel_dim"]]) + 2*TC["linear_num_value_heads"] +
              TC["linear_value_head_dim"] + numel([TC["hidden_size"], vd]))
fa_active = (numel([TC["num_attention_heads"]*TC["head_dim"]*2, TC["hidden_size"]]) +
             2*numel([TC["num_key_value_heads"]*TC["head_dim"], TC["hidden_size"]]) +
             numel([TC["hidden_size"], TC["num_attention_heads"]*TC["head_dim"]]) +
             2*TC["head_dim"])
n_fa = sum(1 for i in range(TC["num_hidden_layers"]) if (i+1) % 4 == 0)
n_gdn = TC["num_hidden_layers"] - n_fa

moe_active = (numel([TC["num_experts"], TC["hidden_size"]]) +
              TC["num_experts_per_tok"] * (numel([2*TC["moe_intermediate_size"], TC["hidden_size"]]) +
                                           numel([TC["hidden_size"], TC["moe_intermediate_size"]])) +
              2*numel([TC["shared_expert_intermediate_size"], TC["hidden_size"]]) +
              numel([TC["hidden_size"], TC["shared_expert_intermediate_size"]]) +
              TC["hidden_size"])
attn_total = n_gdn * gdn_active + n_fa * fa_active
moe_total = TC["num_hidden_layers"] * moe_active
norms_total = buckets["归一化层"] - 2*TC["hidden_size"]  # 减去 MTP 与视觉的 norm（粗略，见下）
# 语言主干逐层 norm：input_layernorm + post_attention_layernorm 各 4096，每层 8192
lang_norm_per_layer = 2 * TC["hidden_size"]
lang_norms = TC["num_hidden_layers"] * lang_norm_per_layer + TC["hidden_size"]  # + 末端 norm
lm_head = buckets["输出头"]
active = attn_total + moe_total + lang_norms + lm_head

print("=" * 66)
print("单 token 激活量核算")
print("=" * 66)
print(f"  GDN 层激活（每层）     {gdn_active:>15,}")
print(f"  全注意力层激活（每层） {fa_active:>15,}")
print(f"  MoE 层激活（每层）     {moe_active:>15,}")
print(f"  注意力合计（{n_gdn} GDN + {n_fa} 全注意力）   {attn_total:>15,}")
print(f"  MoE 合计（{TC['num_hidden_layers']} 层）        {moe_total:>15,}")
print(f"  语言主干 norm 合计     {lang_norms:>15,}")
print(f"  lm_head               {lm_head:>15,}")
print(f"  激活合计（不含 embed 查表） {active:>15,}  ({active/1e9:.3f} B)")
print()
print(f"  对照官方宣称：总参数 397B -> 实算 {total/1e9:.2f} B；激活 17B -> 实算 {active/1e9:.3f} B")
print(f"  激活占比 {active/total*100:.3f}%")
print(f"  含词嵌入查表的激活合计 = {active + buckets['词嵌入']:,}（官方 17B 命名口径）")
print()
print(f"  单层 MoE 激活分解：路由专家 {TC['num_experts_per_tok']*(2*TC['moe_intermediate_size']*TC['hidden_size'] + TC['hidden_size']*TC['moe_intermediate_size']):,}"
      f" + 共享 {3*TC['shared_expert_intermediate_size']*TC['hidden_size']:,}"
      f" + 路由器 {TC['num_experts']*TC['hidden_size']:,}"
      f" + 共享门 {TC['hidden_size']:,}")
print(f"  单层 MoE 激活占全量比例 = {moe_active / (numel([TC['num_experts'], 2*TC['moe_intermediate_size'], TC['hidden_size']]) + numel([TC['num_experts'], TC['hidden_size'], TC['moe_intermediate_size']]) + 3*TC['shared_expert_intermediate_size']*TC['hidden_size'] + TC['num_experts']*TC['hidden_size'] + TC['hidden_size'])*100:.2f}%")
print(f"  视觉塔占全模型比例 = {buckets['视觉塔']/total*100:.3f}%")
print()
print("=" * 66)
print("视觉塔参数分组")
print("=" * 66)
import collections
vg = collections.Counter()
for k, v in H.items():
    if not k.startswith("model.visual."): continue
    n = numel(v["shape"])
    if ".blocks." in k:
        if ".attn." in k: vg["27 层 attention"] += n
        elif ".mlp." in k: vg["27 层 MLP"] += n
        else: vg["27 层 LayerNorm"] += n
    elif k.startswith("model.visual.merger."): vg["merger"] += n
    elif k.startswith("model.visual.pos_embed."): vg["pos_embed"] += n
    elif k.startswith("model.visual.patch_embed."): vg["patch_embed"] += n
vt = sum(vg.values())
for k2, v2 in vg.most_common():
    print(f"  {k2:<20s} {v2:>12,}  {v2/vt*100:6.2f}%")
print(f"  {'合计':<20s} {vt:>12,}")
print()
print(f"  语言主干（不含视觉/MTP）= {total - buckets['视觉塔'] - buckets['MTP 草稿层']:,}")
print(f"  路由专家单层全量 = {numel([TC['num_experts'], 2*TC['moe_intermediate_size'], TC['hidden_size']]) + numel([TC['num_experts'], TC['hidden_size'], TC['moe_intermediate_size']]):,}")
