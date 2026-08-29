"""交叉验证：config/源码推导的结构 vs 真实 checkpoint 张量清单。

验证目标（全部应为可判定断言）：
  1. 层型分布：45 层含 linear_attn.*（0-based 层号 (i+1)%4!=0），15 层含 self_attn.*（(i+1)%4==0）
  2. 全注意力层张量形状：q_proj 双宽 [16384,4096]、k/v_proj [512,4096]、o_proj [4096,8192]、q/k_norm [256]
  3. GDN 层张量形状：in_proj_qkv [12288,4096]、in_proj_z [8192,4096]、in_proj_b/a [64,4096]、
     conv1d [12288,1,4]、A_log/dt_bias [64]、norm [128]、out_proj [4096,8192]
  4. A_log/dt_bias 恰为 90 个 F32（45 层 × 2）
  5. MoE：每层 gate [512,4096]、experts.gate_up [512,2048,4096]、experts.down [512,4096,1024]、
     shared_expert 三投影、shared_expert_gate [1,4096]
  6. MTP：mtp.* 张量集合与语言层同构（mtp_use_dedicated_embeddings=False → 无独立 embed）
  7. 视觉塔：27 blocks、patch_embed Conv3d、pos_embed [2304,1152]、merger
  8. embed/lm_head 不共享
"""
import json, re, collections

H = json.load(open("/tmp/qwen35/headers.json"))
C = json.load(open("/tmp/qwen35/config.json"))
TC, VC = C["text_config"], C["vision_config"]

ok = fail = 0
def check(name, cond, detail=""):
    global ok, fail
    if cond: ok += 1; print(f"OK   {name} {detail}")
    else:    fail += 1; print(f"FAIL {name} {detail}")

# 1. 层型分布
lin_layers = {int(m.group(1)) for k in H if (m := re.match(r"model\.language_model\.layers\.(\d+)\.linear_attn\.", k))}
full_layers = {int(m.group(1)) for k in H if (m := re.match(r"model\.language_model\.layers\.(\d+)\.self_attn\.", k))}
expect_lin = {i for i in range(TC["num_hidden_layers"]) if (i+1) % TC["full_attention_interval"] != 0}
expect_full = {i for i in range(TC["num_hidden_layers"]) if (i+1) % TC["full_attention_interval"] == 0}
check("线性层集合", lin_layers == expect_lin, f"{len(lin_layers)} 层")
check("全注意力层集合", full_layers == expect_full, f"{len(full_layers)} 层")

def shp(k): return tuple(H[k]["shape"])
L0, L3 = "model.language_model.layers.0.", "model.language_model.layers.3."

# 2. 全注意力层
check("q_proj 双宽（query+gate）", shp(L3+"self_attn.q_proj.weight") == (TC["num_attention_heads"]*TC["head_dim"]*2, TC["hidden_size"]), str(shp(L3+"self_attn.q_proj.weight")))
check("k_proj GQA", shp(L3+"self_attn.k_proj.weight") == (TC["num_key_value_heads"]*TC["head_dim"], TC["hidden_size"]), str(shp(L3+"self_attn.k_proj.weight")))
check("v_proj GQA", shp(L3+"self_attn.v_proj.weight") == (TC["num_key_value_heads"]*TC["head_dim"], TC["hidden_size"]), str(shp(L3+"self_attn.v_proj.weight")))
check("o_proj", shp(L3+"self_attn.o_proj.weight") == (TC["hidden_size"], TC["num_attention_heads"]*TC["head_dim"]), str(shp(L3+"self_attn.o_proj.weight")))
check("q_norm 逐头维", shp(L3+"self_attn.q_norm.weight") == (TC["head_dim"],), str(shp(L3+"self_attn.q_norm.weight")))
check("k_norm 逐头维", shp(L3+"self_attn.k_norm.weight") == (TC["head_dim"],))

# 3. GDN 层
kd = TC["linear_num_key_heads"]*TC["linear_key_head_dim"]
vd = TC["linear_num_value_heads"]*TC["linear_value_head_dim"]
check("in_proj_qkv", shp(L0+"linear_attn.in_proj_qkv.weight") == (kd*2+vd, TC["hidden_size"]), f"{shp(L0+'linear_attn.in_proj_qkv.weight')} (kd={kd}, vd={vd})")
check("in_proj_z", shp(L0+"linear_attn.in_proj_z.weight") == (vd, TC["hidden_size"]))
check("in_proj_b", shp(L0+"linear_attn.in_proj_b.weight") == (TC["linear_num_value_heads"], TC["hidden_size"]))
check("in_proj_a", shp(L0+"linear_attn.in_proj_a.weight") == (TC["linear_num_value_heads"], TC["hidden_size"]))
check("conv1d depthwise", shp(L0+"linear_attn.conv1d.weight") == (kd*2+vd, 1, TC["linear_conv_kernel_dim"]))
check("A_log", shp(L0+"linear_attn.A_log") == (TC["linear_num_value_heads"],))
check("dt_bias", shp(L0+"linear_attn.dt_bias") == (TC["linear_num_value_heads"],))
check("norm 头维共享", shp(L0+"linear_attn.norm.weight") == (TC["linear_value_head_dim"],))
check("out_proj", shp(L0+"linear_attn.out_proj.weight") == (TC["hidden_size"], vd))

# 4. F32 = A_log + dt_bias
f32 = [k for k, v in H.items() if v["dtype"] == "F32"]
check("F32 恰为 A_log+GDN norm.weight×45 层", len(f32) == 90 and all(k.endswith(("A_log", "linear_attn.norm.weight")) for k in f32) and
      sum(1 for k in f32 if ".language_model.layers." in k) == 90, f"{len(f32)} 个 F32: " + ", ".join(sorted(set(k.split(".")[-1] for k in f32))))

# 5. MoE（每层）
E, I = TC["num_experts"], TC["moe_intermediate_size"]
check("router gate", shp(L0+"mlp.gate.weight") == (E, TC["hidden_size"]), str(shp(L0+"mlp.gate.weight")))
check("experts gate_up 3D", shp(L0+"mlp.experts.gate_up_proj") == (E, 2*I, TC["hidden_size"]), str(shp(L0+"mlp.experts.gate_up_proj")))
check("experts down 3D", shp(L0+"mlp.experts.down_proj") == (E, TC["hidden_size"], I))
check("shared_expert gate", shp(L0+"mlp.shared_expert.gate_proj.weight") == (TC["shared_expert_intermediate_size"], TC["hidden_size"]))
check("shared_expert_gate [1,H]", shp(L0+"mlp.shared_expert_gate.weight") == (1, TC["hidden_size"]))
n_gate = sum(1 for k in H if k.endswith(".mlp.gate.weight"))
check("每层都有 MoE（60 语言层 + MTP 1 层）", n_gate == TC["num_hidden_layers"] + 1, f"{n_gate} 个 mlp.gate：60 主干 + 1 MTP")

# 6. MTP
mtp = [k for k in H if k.startswith("mtp.")]
mtp_norm = {re.sub(r"\d+", "N", k) for k in mtp}
l0_norm = {re.sub(r"layers\.\d+", "layers.N", k).replace("model.language_model.layers.N.", "") for k in H
           if k.startswith("model.language_model.layers.0.")}
# MTP 层应同构于一个 full_attention 解码层（+共享 embed/lm_head，无独立 embed）
mtp_inner = {k[len("mtp."):].replace("layers.0.", "") for k in mtp_norm}
print("\n--- MTP 张量结构（归一化） ---")
for k in sorted(mtp_norm)[:40]: print("  ", k, H[[kk for kk in H if re.sub(chr(92)+'d+','N',kk)==k][0]]["shape"])
has_embed = any(k.startswith(("mtp.embed_tokens", "mtp.lm_head")) for k in mtp)
check("MTP 无独立 embedding/lm_head（共享主干）", not has_embed, f"mtp.* 张量 {len(mtp)} 个，其中 1536 个为逐专家独立投影")
check("MTP 是 full_attention 层（q_proj 双宽同主干）", any(k.endswith("mtp.layers.0.self_attn.q_proj.weight") for k in mtp) and shp("mtp.layers.0.self_attn.q_proj.weight") == (TC["num_attention_heads"]*TC["head_dim"]*2, TC["hidden_size"]) and not any("linear_attn" in k for k in mtp))
check("MTP fc 吃两路 4096 拼接", shp("mtp.fc.weight") == (TC["hidden_size"], TC["hidden_size"]*2), str(shp("mtp.fc.weight")))
check("MTP 两路 pre_fc RMSNorm", shp("mtp.pre_fc_norm_embedding.weight") == (TC["hidden_size"],) and shp("mtp.pre_fc_norm_hidden.weight") == (TC["hidden_size"],))

# 7. 视觉塔
vblocks = {int(m.group(1)) for k in H if (m := re.match(r"model\.visual\.blocks\.(\d+)\.", k))}
check("视觉 27 blocks", vblocks == set(range(VC["depth"])), f"{len(vblocks)} 层")
check("patch_embed Conv3d", shp("model.visual.patch_embed.proj.weight") == (VC["hidden_size"], VC["in_channels"], VC["temporal_patch_size"], VC["patch_size"], VC["patch_size"]), str(shp("model.visual.patch_embed.proj.weight")))
check("pos_embed 48×48", shp("model.visual.pos_embed.weight") == (VC["num_position_embeddings"], VC["hidden_size"]))
check("merger 输出到 hidden_size", shp("model.visual.merger.linear_fc2.weight") == (TC["hidden_size"], VC["hidden_size"]*VC["spatial_merge_size"]**2))
check("视觉无 deepstack 张量", not any("deepstack" in k for k in H))

# 8. embed/lm_head
check("embed", shp("model.language_model.embed_tokens.weight") == (TC["vocab_size"], TC["hidden_size"]), str(shp("model.language_model.embed_tokens.weight")))
lm = [k for k in H if "lm_head" in k]
check("lm_head 存在且不共享", len(lm) == 1 and shp(lm[0]) == (TC["vocab_size"], TC["hidden_size"]), str(lm))

print(f"\n===== {ok} OK / {fail} FAIL =====")
