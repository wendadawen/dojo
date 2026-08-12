"""实测六：从真实 checkpoint 读取张量元数据，核对形状与 dtype。

safetensors 文件头是 JSON，位于文件开头（前 8 字节为头长度，小端 u64）。
用 HTTP Range 请求只取文件头，不下载 864 GB 权重本体。
"""
import json, subprocess, struct

REPO = "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main"
idx = json.load(open("/tmp/dsv4/index.json"))
wm = idx["weight_map"]

# 目标张量：覆盖数据流每一级
targets = [
    "embed.weight",
    "layers.0.attn.wq_a.weight", "layers.0.attn.wq_b.weight",
    "layers.0.attn.wkv.weight", "layers.0.attn.kv_norm.weight",
    "layers.0.attn.wo_a.weight", "layers.0.attn.wo_b.weight",
    "layers.0.attn.attn_sink",
    "layers.0.attn.compressor.wkv.weight", "layers.0.attn.compressor.wgate.weight",
    "layers.0.attn.compressor.ape",
    "layers.0.hc_attn_fn", "layers.0.hc_attn_base", "layers.0.hc_attn_scale",
    "layers.0.ffn.gate.weight", "layers.0.ffn.gate.tid2eid",
    "layers.0.ffn.experts.0.w1.weight", "layers.0.ffn.experts.0.w1.scale",
    "layers.0.ffn.shared_experts.w1.weight",
    "layers.2.attn.indexer.wq_b.weight", "layers.2.attn.indexer.weights_proj.weight",
    "layers.2.attn.indexer.compressor.wkv.weight", "layers.2.attn.indexer.compressor.ape",
    "layers.2.attn.compressor.wkv.weight",
    "layers.3.ffn.gate.bias",
    "head.weight", "norm.weight",
    "mtp.0.e_proj.weight", "mtp.0.h_proj.weight", "mtp.0.hc_head_fn",
]

files = sorted({wm[t] for t in targets if t in wm})
headers = {}
for f in files:
    url = f"{REPO}/{f}"
    # 先取前 8 字节得到头长度
    r = subprocess.run(["curl", "-sL", "-r", "0-7", url], capture_output=True)
    n = struct.unpack("<Q", r.stdout[:8])[0]
    r2 = subprocess.run(["curl", "-sL", "-r", f"8-{8+n-1}", url], capture_output=True)
    headers[f] = json.loads(r2.stdout.decode("utf-8"))
    print(f"读取 {f} 头部 {n} 字节，含 {len(headers[f])-1} 个张量")

print()
print(f"{'张量名':<48} {'dtype':<10} {'形状'}")
print("-" * 90)
for t in targets:
    if t not in wm:
        print(f"{t:<48} {'(不存在)'}")
        continue
    h = headers[wm[t]]
    if t not in h:
        print(f"{t:<48} {'(头中未找到)'}")
        continue
    e = h[t]
    print(f"{t:<48} {e['dtype']:<10} {e['shape']}")
