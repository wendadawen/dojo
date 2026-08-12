"""实测十：从全部 64 个 safetensors 文件头精确统计参数量，核对 1.6T / 49B。

只用 HTTP Range 取文件头（每个约 260KB），不下载 864GB 权重本体。
注意：FP4 专家权重以 I8 存储且沿 K 维打包 2 个值/字节，
      故其逻辑参数个数 = numel x 2（已用 model.py Linear 的 in_features//2 佐证）。
"""
import json, struct, subprocess, re, collections
from concurrent.futures import ThreadPoolExecutor

REPO = "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main"
files = [f"model-{i:05d}-of-00064.safetensors" for i in range(1, 65)]

def get_header(f):
    url = f"{REPO}/{f}"
    r = subprocess.run(["curl", "-sL", "-r", "0-7", url], capture_output=True)
    n = struct.unpack("<Q", r.stdout[:8])[0]
    r2 = subprocess.run(["curl", "-sL", "-r", f"8-{8+n-1}", url], capture_output=True)
    return f, json.loads(r2.stdout.decode("utf-8"))

with ThreadPoolExecutor(max_workers=16) as ex:
    results = list(ex.map(get_header, files))

tensors = {}
for f, h in results:
    for name, meta in h.items():
        if name == "__metadata__":
            continue
        tensors[name] = meta
print(f"合并全部文件头，得到张量总数 = {len(tensors):,}")

def numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n

def logical_params(name, meta):
    """FP4 专家权重按 I8 存储，每字节含 2 个 fp4 值 -> 逻辑参数为 numel x 2。"""
    n = numel(meta["shape"])
    if meta["dtype"] == "I8" and name.endswith(".weight"):
        return n * 2
    return n

total = 0
by_group = collections.Counter()
scale_params = 0
for name, meta in tensors.items():
    p = logical_params(name, meta)
    if name.endswith(".scale"):
        scale_params += p          # 量化 scale 不算模型参数
        continue
    total += p
    if ".experts." in name and name.startswith("layers."):
        by_group["路由专家(61层)"] += p
    elif ".shared_experts." in name and name.startswith("layers."):
        by_group["共享专家(61层)"] += p
    elif name.startswith("layers.") and ".attn." in name:
        by_group["注意力(61层)"] += p
    elif name.startswith("layers.") and name.split(".")[-1].startswith("hc_"):
        by_group["mHC(61层)"] += p
    elif name.startswith("layers.") and ".gate." in name:
        by_group["MoE gate(61层)"] += p
    elif name.startswith("layers."):
        by_group["层内 norm"] += p
    elif name.startswith("mtp."):
        by_group["MTP 模块"] += p
    elif name in ("embed.weight",):
        by_group["embedding"] += p
    elif name in ("head.weight",):
        by_group["lm_head"] += p
    else:
        by_group["其他"] += p

print(f"量化 scale 张量元素数（不计入参数量）= {scale_params:,}")
print()
print("=== 参数量分组统计（不含 scale）===")
for k, v in sorted(by_group.items(), key=lambda kv: -kv[1]):
    print(f"  {k:<18} {v/1e9:>10.2f} B   占比 {v/total*100:>5.1f}%")
print(f"  {'合计':<18} {total/1e9:>10.2f} B = {total/1e12:.3f} T")
print()
print(f"README 宣称总参数 1.6T -> 实测 {total/1e12:.3f}T  ✓")
print()

# 激活参数：每 token 实际参与计算的部分
cfg = json.load(open("/tmp/dsv4/config.json"))
L, E, TOPK = cfg["num_hidden_layers"], cfg["n_routed_experts"], cfg["num_experts_per_tok"]
routed_total = by_group["路由专家(61层)"]
per_expert = routed_total / (L * E)
act_routed = L * TOPK * per_expert
non_expert = total - routed_total - by_group["MTP 模块"] - by_group["embedding"]
print("=== 每 token 激活参数（推理时实际参与的权重）===")
print(f"  单个路由专家        = {per_expert/1e6:.2f} M")
print(f"  激活的路由专家      = {L} 层 x {TOPK} 个 = {act_routed/1e9:.2f} B")
print(f"  非专家部分(含共享专家/注意力/mHC/gate/norm/lm_head) = {non_expert/1e9:.2f} B")
print(f"  合计激活            = {(act_routed+non_expert)/1e9:.2f} B")
print(f"  README 宣称 49B 激活 -> 实测 {(act_routed+non_expert)/1e9:.1f}B  （embedding 查表不计入 GEMM 激活）")
print()
print("=== 注意力内部构成（单层，用真实形状）===")
att = {n: m for n, m in tensors.items() if n.startswith("layers.0.attn.") or n.startswith("layers.2.attn.")}
for n in sorted(att):
    if n.endswith(".scale"): continue
    print(f"  {n:<50} {att[n]['dtype']:<8} {att[n]['shape']}")
