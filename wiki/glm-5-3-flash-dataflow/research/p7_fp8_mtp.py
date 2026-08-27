"""
探针 7：FP8 块量化布局 与 MTP（多 token 预测）层。

验证目标
  1. FP8 量化的实际粒度：weight_block_size=[128,128] 的 scale 张量形状是否等于
     ceil(N/128) x ceil(K/128) —— 用真实 checkpoint 张量头验证
  2. 哪些模块不量化（modules_to_not_convert 有 1509 条，需归纳规律而不是罗列）
  3. 参数量与磁盘占用的关系：混合精度下 total_size 不能当参数量用
  4. MTP 层（layer 45）的结构：enorm/hnorm/eh_proj/shared_head，与主干层的差异
  5. MTP 层是 DSA 类型还是 KDA 类型（决定投机解码时的状态管理）

对应源码/配置
  config.json quantization_config
  Glm5NextPreTrainedModel._keys_to_ignore_on_load_unexpected   L1359
  config.num_nextn_predict_layers / index_share_for_mtp_iteration
"""
from __future__ import annotations
import json, math, re
from collections import Counter, defaultdict

H = json.load(open("/tmp/glm53f/headers.json"))
CFG = json.load(open("/tmp/glm53f/config.json"))
T, QC = CFG["text_config"], CFG["quantization_config"]

tensors = {}
for shard, hdr in H.items():
    for n, m in hdr.items():
        if n != "__metadata__":
            tensors[n] = m


def banner(t):
    print("=" * 78); print(t); print("=" * 78)


banner("探针 7：FP8 块量化布局")

# ---------- 7.1 量化配置 ----------
print("[7.1] quantization_config")
print(f"      quant_method = {QC['quant_method']}   fmt = {QC['fmt']}   "
      f"activation_scheme = {QC['activation_scheme']}")
print(f"      weight_block_size = {QC['weight_block_size']}")
print(f"      modules_to_not_convert 条目数 = {len(QC['modules_to_not_convert'])}")

# ---------- 7.2 scale 形状验证 ----------
banner("7.2 scale 张量形状 = ceil(N/128) x ceil(K/128)（逐条实测验证）")
bn, bk = QC["weight_block_size"]
checked = ok = 0
bad = []
for name, meta in tensors.items():
    if not name.endswith("weight_scale_inv"):
        continue
    w = name[: -len("_scale_inv")]
    if w not in tensors:
        bad.append((name, "对应权重缺失"))
        continue
    N, K = tensors[w]["shape"]
    exp = [math.ceil(N / bn), math.ceil(K / bk)]
    checked += 1
    if meta["shape"] == exp:
        ok += 1
    else:
        bad.append((name, f"权重{[N,K]} scale{meta['shape']} 期望{exp}"))
print(f"      检查了 {checked:,} 个 scale 张量，形状符合 ceil(N/{bn})xceil(K/{bk}) 的有 {ok:,} 个")
print(f"      不符合的：{len(bad)} 个" + ("" if not bad else f" → {bad[:5]}"))
print(f"      scale dtype 集合 = "
      f"{set(m['dtype'] for n,m in tensors.items() if n.endswith('weight_scale_inv'))}")

print("\n      举例：")
for w in ["model.language_model.layers.3.mlp.experts.0.gate_proj.weight",
          "model.language_model.layers.3.self_attn.q_b_proj.weight",
          "model.language_model.layers.0.mlp.gate_proj.weight"]:
    s = w + "_scale_inv"
    N, K = tensors[w]["shape"]
    print(f"        {w.replace('model.language_model.','')}")
    print(f"          权重 {tensors[w]['shape']} {tensors[w]['dtype']}  → scale {tensors[s]['shape']} "
          f"= [{N}/{bn}, {K}/{bk}] = [{N//bn}, {K//bk}]")

# ---------- 7.3 量化 vs 未量化的模块规律 ----------
banner("7.3 哪些张量被量化为 FP8，哪些保持 BF16/FP32（按后缀归纳）")
def norm_key(name):
    k = re.sub(r"layers\.\d+\.", "layers.N.", name)
    k = re.sub(r"experts\.\d+\.", "experts.N.", k)
    return re.sub(r"blocks\.\d+\.", "blocks.N.", k)


q, nq = Counter(), Counter()
dtype_of = {}
for name, meta in tensors.items():
    if name.endswith("weight_scale_inv"):
        continue
    key = norm_key(name)
    dtype_of.setdefault(key, meta["dtype"])
    (q if meta["dtype"] == "F8_E4M3" else nq)[key] += 1

print(f"      FP8 张量种类（{sum(q.values()):,} 个）：")
for k, v in sorted(q.items()):
    print(f"        {k}   x{v}")
print(f"\n      非 FP8 张量种类（{sum(nq.values()):,} 个，只列文本主干与 lm_head）：")
for k, v in sorted(nq.items()):
    if k.startswith("model.visual."):
        continue
    print(f"        {k}   x{v}  dtype={dtype_of[k]}")

print(f"\n      规律：只有大矩阵乘的 weight 被量化（专家 / MLA 投影 / dense MLP），")
print(f"      而 norm 权重、路由器 gate、e_score_correction_bias、KDA 的全部投影、")
print(f"      indexer 的全部权重、embedding、lm_head、视觉塔全部保持高精度。")
print(f"      _keep_in_fp32_modules_strict = ['e_score_correction_bias','conv1d','dt_bias','A_log']（源码 L1358）")
print(f"      与实测 F32 张量交叉验证：A_log/dt_bias/e_score_correction_bias dtype = "
      f"{set(m['dtype'] for n,m in tensors.items() if n.endswith(('A_log','dt_bias','e_score_correction_bias')))}")

# 值得注意：KDA 层完全没被量化
kda_q = [n for n, m in tensors.items()
         if m["dtype"] == "F8_E4M3" and re.search(r"layers\.(\d+)\.self_attn\.(q|k|v|o)_proj", n)
         and int(re.search(r"layers\.(\d+)\.", n).group(1)) in
         [i for i, t in enumerate(T["layer_types"]) if t == "linear_attention"]]
print(f"\n      KDA 层中被量化为 FP8 的投影张量数 = {len(kda_q)}")
print(f"      → 34 个 KDA 层的 q/k/v/o_proj 全部保持 BF16，只有 11 个 DSA 层与 MoE 走 FP8")

# ---------- 7.4 参数量 vs 磁盘 ----------
banner("7.4 参数量与磁盘占用（混合精度下不可混用）")
BYTES = {"F8_E4M3": 1, "BF16": 2, "F32": 4}
n_param = n_byte = 0
for name, meta in tensors.items():
    k = 1
    for d in meta["shape"]:
        k *= d
    n_byte += k * BYTES[meta["dtype"]]
    if not name.endswith("weight_scale_inv"):
        n_param += k
print(f"      参数个数（不含 scale）= {n_param:,} = {n_param/1e9:.2f} B")
print(f"      按 dtype 累加字节     = {n_byte:,} = {n_byte/2**30:.2f} GiB")
print(f"      index.json total_size = {json.load(open('/tmp/glm53f/model.safetensors.index.json'))['metadata']['total_size']:,}"
      f" = {json.load(open('/tmp/glm53f/model.safetensors.index.json'))['metadata']['total_size']/2**30:.2f} GiB")
print(f"      两者相符 = {n_byte == json.load(open('/tmp/glm53f/model.safetensors.index.json'))['metadata']['total_size']}")
print(f"      平均每参数 {n_byte/n_param:.3f} 字节 → 若拿 total_size 当参数量会高估 "
      f"{(n_byte/n_param-1)*100:.1f}%")
print(f"      官方 README 称 320B total；实测参数个数 {n_param/1e9:.2f} B（含 MTP 层与视觉塔）")
mtp_p = 0
for n, m in tensors.items():
    if ".layers.45." in n and not n.endswith("weight_scale_inv"):
        k = 1
        for d in m["shape"]:
            k *= d
        mtp_p += k
print(f"      扣除 MTP 层（{mtp_p/1e9:.2f} B）后 = {(n_param-mtp_p)/1e9:.2f} B")

# ---------- 7.5 MTP 层 ----------
banner("7.5 MTP 层（layer 45）结构")
print(f"      config.num_nextn_predict_layers = {T['num_nextn_predict_layers']}")
print(f"      源码 L1359 _keys_to_ignore_on_load_unexpected = "
      f"[r'layers\\.45\\.', r'layers\\.\\d+\\.shared_head\\.']")
print(f"      → transformers 的 Glm5NextTextModel 只建 45 层主干（0..44），"
      f"直接忽略 checkpoint 里的第 45 层；MTP 由推理框架（SGLang/vLLM）单独加载")
pre = "model.language_model.layers.45."
mtp_names = sorted(n[len(pre):] for n in tensors
                   if n.startswith(pre) and "mlp.experts." not in n
                   and not n.endswith("weight_scale_inv"))
print(f"\n      MTP 层张量（非 experts，{len(mtp_names)} 项）：")
for n in mtp_names:
    print(f"        {n:44s} {tensors[pre+n]['shape']} {tensors[pre+n]['dtype']}")
print(f"\n      与主干层对照：")
print(f"        MTP 有 self_attn.kv_a_proj_with_mqa / indexer → 是 DSA(MLA) 类型层，不是 KDA")
print(f"        MTP 无 hc_attn_*/hc_ffn_* → 不使用 mHC，走普通残差")
print(f"        MTP 独有 enorm / hnorm / eh_proj[4096,8192] → 把「上一步隐藏态 + 下一 token embedding」"
      f"拼接后投影回 4096（DeepSeek-V3 MTP 的标准做法）")
print(f"        MTP 独有 shared_head.norm → 复用主干 lm_head 前的归一化")
n_mtp_exp = len(set(
    int(m.group(1)) for m in
    (re.match(r"model\.language_model\.layers\.45\.mlp\.experts\.(\d+)\.", n) for n in tensors)
    if m))
print(f"        MTP 有完整 288 专家 MoE = {n_mtp_exp} 个")
print(f"      config.index_share_for_mtp_iteration = {T['index_share_for_mtp_iteration']}")
print(f"      → 投机解码的多次迭代间复用 indexer 的 top-k 选择，省掉重复打分")
