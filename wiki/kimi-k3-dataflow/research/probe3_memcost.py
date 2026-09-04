# 实测三：长上下文推理开销（缓存/状态体积），按官方 config.json 真实维度精确计算。
# 口径：权重与缓存均按 bf16（2 字节）计。
#
# MLA 层两个口径分开：
#   A. 官方 HF 参考实现：cache 的是 kv_b_proj 展开后的 96 头 K(192) + V(128)
#      （modeling_kimi_linear.py forward: key_states=cat(k_pass,k_rot) 后 update）
#   B. 潜空间压缩口径（生产部署可用）：每 token 只存压缩潜向量 ckv(512) + k_rot(64) = 576
#      （kv_b_proj 权重可吸收进 q/k 投影，DeepSeek-V3 MLA 的标准做法）
GiB = 1024 ** 3
MiB = 1024 ** 2
KiB = 1024

H = 96            # num_attention_heads
QK_NOPE = 128     # qk_nope_head_dim
QK_ROT = 64       # qk_rope_head_dim
V_DIM = 128       # v_head_dim
KV_LORA = 512     # kv_lora_rank
N_MLA = 24        # full_attn_layers 数
N_KDA = 69
N_ALL = 93
KDA_HEADS = 96
KDA_KDIM = 128
KDA_VDIM = 128
CONV_CH = 12288   # 96*128，q/k/v 三个卷积通道数
CONV_K = 4        # short_conv_kernel_size

mla_per_tok_ref = H * (QK_NOPE + QK_ROT + V_DIM) * 2      # 展开 K(192)+V(128)
mla_per_tok_latent = (KV_LORA + QK_ROT) * 2               # ckv 512 + k_rot 64
kda_state_fixed = (KDA_HEADS * KDA_KDIM * KDA_VDIM + 3 * CONV_CH * (CONV_K - 1)) * 2

print(f"MLA 参考实现口径: {mla_per_tok_ref:,} B/token/层 = {mla_per_tok_ref/KiB:.1f} KiB/token/层 (96*(192+128)*2B)")
print(f"MLA 潜压缩口径:   {mla_per_tok_latent:,} B/token/层 = {mla_per_tok_latent/KiB:.3f} KiB/token/层 ((512+64)*2B)")
print(f"KDA 固定状态/层:  递归 {KDA_HEADS*KDA_KDIM*KDA_VDIM*2/KiB:.1f} KiB + 卷积 {3*CONV_CH*(CONV_K-1)*2/KiB:.1f} KiB = {kda_state_fixed/KiB:.1f} KiB")
print(f"KDA 69 层合计: {N_KDA*kda_state_fixed/MiB:.1f} MiB = {N_KDA*kda_state_fixed/GiB:.3f} GiB（与序列长度无关）")
print()
print(f"{'上下文':>8} | {'MLA KV×24层(参考实现)':>22} | {'MLA KV×24层(潜压缩)':>20} | {'KDA 状态×69层':>14} | {'合计(参考口径)':>14}")
for name, n in [("4K", 4096), ("32K", 32768), ("128K", 131072), ("256K", 262144), ("1M", 1048576)]:
    kv_ref = N_MLA * n * mla_per_tok_ref
    kv_lat = N_MLA * n * mla_per_tok_latent
    kda = N_KDA * kda_state_fixed
    tot = kv_ref + kda
    print(f"{name:>8} | {kv_ref/GiB:>18.3f} GiB | {kv_lat/GiB:>16.3f} GiB | {kda/GiB:>10.3f} GiB | {tot/GiB:>10.3f} GiB")

print()
print("== 层型比例的意义（若 93 层全为 MLA 参考实现口径，1M 上下文） ==")
all_mla = N_ALL * 1048576 * mla_per_tok_ref
print(f"93 层全 MLA: {all_mla/GiB:.1f} GiB = {all_mla/(1024*GiB):.2f} TiB")
print(f"实际 24 层 MLA: {24*1048576*mla_per_tok_ref/GiB:.1f} GiB -> 降到 {24/93*100:.1f}%（KDA 69 层只贡献固定 {N_KDA*kda_state_fixed/GiB:.3f} GiB）")
print()
print("== 潜压缩口径下 1M 上下文（生产部署 MLA 吸收后） ==")
print(f"24 层潜压缩 KV: {24*1048576*mla_per_tok_latent/GiB:.2f} GiB + KDA {N_KDA*kda_state_fixed/GiB:.3f} GiB = {(24*1048576*mla_per_tok_latent + N_KDA*kda_state_fixed)/GiB:.2f} GiB")
print()
print("== 每层 token 开支对照 ==")
print(f"一个视觉 token(7168 宽 bf16 embedding) = {7168*2/KiB:.1f} KiB；"
      f"一张 448x448 图 256 视觉 token 进入主干后，MLA 参考口径新增 KV {256*24*mla_per_tok_ref/MiB:.1f} MiB（prefill 一次性）")
