"""交叉验证：用真实 checkpoint 张量清单核对源码推导的结构性结论。

依据来源：
- config: Qwen/Qwen3.8-Flash-Next@f5d08274 config.json
- 源码: transformers@36deb0b5 src/transformers/models/qwen4_exp/modeling_qwen4_exp.py
"""
import json, re, collections

H = json.load(open("/tmp/qwen38fn/headers.json"))
CFG = json.load(open("/tmp/qwen38fn/config.json"))
TC = CFG["text_config"]

print("=" * 70)
print("A. 顶层模块前缀分布")
print("=" * 70)
top = collections.Counter()
for k in H:
    top[".".join(k.split(".")[:2])] += 1
for k, v in sorted(top.items(), key=lambda x: -x[1])[:20]:
    print(f"  {v:6d}  {k}")

print()
print("=" * 70)
print("B. 层类型交叉验证：linear_attn vs self_attn 的层号集合")
print("=" * 70)
lin_layers, qsa_layers, ple_layers = set(), set(), set()
for k in H:
    m = re.match(r"model\.language_model\.layers\.(\d+)\.(\w+)", k)
    if not m:
        continue
    li, mod = int(m.group(1)), m.group(2)
    if mod == "linear_attn":
        lin_layers.add(li)
    elif mod == "self_attn":
        qsa_layers.add(li)
    elif mod == "ple":
        ple_layers.add(li)

declared = TC["layer_types"]
d_lin = {i for i, t in enumerate(declared) if t == "linear_attention"}
d_full = {i for i, t in enumerate(declared) if t == "full_attention"}
print(f"  config 声明 linear_attention 层数: {len(d_lin)}")
print(f"  权重出现 linear_attn 的层数    : {len(lin_layers)}   一致={d_lin == lin_layers}")
print(f"  config 声明 full_attention 层数 : {len(d_full)}")
print(f"  权重出现 self_attn 的层数      : {len(qsa_layers)}   一致={d_full == qsa_layers}")
print(f"  full_attention 层号(0-based)   : {sorted(d_full)}")
print(f"  config ple_layer_ids (1-based) : {TC['ple_layer_ids']}  -> 0-based {[i-1 for i in TC['ple_layer_ids']]}")
print(f"  权重出现 ple 的层号(0-based)   : {sorted(ple_layers)}")

print()
print("=" * 70)
print("C. 每个 QSA 层是否都带 indexer（源码 Attention.__init__ 无条件建 indexer）")
print("=" * 70)
idx_layers = {int(m.group(1)) for k in H
              if ".self_attn.indexer." in k
              and (m := re.match(r"model\.language_model\.layers\.(\d+)\.", k))}
print(f"  含 indexer 的全部张量键前缀: {sorted({re.sub(r'[0-9]+', 'N', k.split('.indexer.')[0]) for k in H if '.indexer.' in k})}")
print(f"  带 indexer 的层号: {sorted(idx_layers)}")
print(f"  与 full_attention 层号一致: {idx_layers == d_full}")
for k in sorted(k for k in H if ".layers.3.self_attn." in k):
    print(f"    {k:70s} {H[k]['dtype']:>6s} {H[k]['shape']}")

print()
print("=" * 70)
print("D. Gated Residual（超连接）张量：每层两组 + 全局 mixer")
print("=" * 70)
hc_names = collections.Counter()
first = {}
for k in H:
    if "hyper_connection" in k:
        key = re.sub(r"layers\.\d+\.", "layers.N.", k)
        hc_names[key] += 1
        first.setdefault(key, k)
for k, v in sorted(hc_names.items()):
    print(f"  x{v:3d}  {k:78s} {H[first[k]]['shape']}")

print()
print("=" * 70)
print("E. PLE / N-gram embedding 张量")
print("=" * 70)
ngram_shards = [k for k in H if "ngram_embedding" in k]
print(f"  ngram_embedding 分片张量数: {len(ngram_shards)}  (config split_ngram_parts={TC['split_ngram_parts']})")
print(f"  样例: {ngram_shards[0]} -> {H[ngram_shards[0]]}")
tot_rows = sum(H[k]["shape"][0] for k in ngram_shards)
cols = {tuple(H[k]["shape"][1:]) for k in ngram_shards}
print(f"  分片行数之和={tot_rows}  列维度集合={cols}")
for k in sorted(k for k in H if ".ple." in k and "ngram_embedding" not in k):
    print(f"    {k:72s} {H[k]['dtype']:>6s} {H[k]['shape']}")

print()
print("=" * 70)
print("F. MoE 张量与 linear_attn 张量样例（第 0 层）")
print("=" * 70)
for k in sorted(k for k in H if ".layers.0." in k):
    print(f"    {k:74s} {H[k]['dtype']:>6s} {H[k]['shape']}")

print()
print("=" * 70)
print("G. MTP / 顶层其他张量")
print("=" * 70)
for k in sorted(H):
    if not re.match(r"model\.language_model\.layers\.\d+\.", k) and not k.startswith("model.visual"):
        print(f"    {k:74s} {H[k]['dtype']:>6s} {H[k]['shape']}")

print()
print("=" * 70)
print("H. dtype 分布")
print("=" * 70)
for d, c in collections.Counter(v["dtype"] for v in H.values()).items():
    print(f"  {d}: {c} 个张量")
