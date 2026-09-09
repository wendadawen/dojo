# 验证目标：按 data_offsets 用 HTTP Range 精确下载 checkpoint 小张量实值，
# 与源码 _init_weights 的初始值对照（训练后偏移多大），并留档供页面引用。
# 目标张量：learnable_sink_param / e_score_correction_bias / hc base+scale / hc_head / k_norm
import json
import struct
import subprocess
from pathlib import Path

REPO = "https://huggingface.co/tencent/Hy4-preview/resolve/main"
HERE = Path(__file__).parent
H = json.loads((HERE / "headers.json").read_text())


def curl_range(url, start, end):
    r = subprocess.run(["curl", "-sL", "-r", f"{start}-{end}", url], capture_output=True)
    return r.stdout


_hlen_cache = {}


def shard_hlen(shard):
    if shard not in _hlen_cache:
        url = f"{REPO}/{shard}"
        _hlen_cache[shard] = struct.unpack("<Q", curl_range(url, 0, 7))[0]
    return _hlen_cache[shard]


def fetch(name):
    meta = H[name]
    url = f"{REPO}/{meta['shard']}"
    hlen = shard_hlen(meta["shard"])
    s, e = meta["data_offsets"]
    raw = curl_range(url, 8 + hlen + s, 8 + hlen + e - 1)
    dt, shape = meta["dtype"], meta["shape"]
    n = (e - s) // (8 if dt in ("F64", "I64") else 4 if dt in ("F32", "I32") else 2 if dt == "BF16" else 1)
    fmt = {"F32": "f", "I32": "i", "BF16": None}.get(dt)
    if dt == "BF16":
        # bf16 = f32 高 16 位：位模式左移 16 后按 f32 重解释
        u16 = struct.unpack(f"<{n}H", raw)
        vals = [struct.unpack("<f", struct.pack("<I", u << 16))[0] for u in u16]
    else:
        vals = list(struct.unpack(f"<{n}{fmt}", raw))
    return vals, shape


def show(name, note=""):
    vals, shape = fetch(name)
    vs = [f"{v:.6g}" for v in vals[:12]]
    print(f"\n{name}  shape={shape}")
    print(f"  first 12: {vs}")
    if len(vals) > 12:
        import statistics
        print(f"  n={len(vals)} min={min(vals):.6g} max={max(vals):.6g} "
              f"mean={statistics.mean(vals):.6g} median={statistics.median(vals):.6g}")
    if note:
        print(f"  note: {note}")


# 1) sink：源码初始 learnable_sink_init=0.0，训练后实际值
show("model.layers.0.self_attn.learnable_sink_param", "源码 init=0.0")
show("model.layers.40.self_attn.learnable_sink_param", "中间层对照")
show("model.layers.77.self_attn.learnable_sink_param", "最后一层对照")

# 2) e_score_correction_bias：源码 init=0，aux-loss-free 偏置训练后值
show("model.layers.1.mlp.gate.e_score_correction_bias", "源码 init=0")
show("model.layers.40.mlp.gate.e_score_correction_bias", "中间层对照")
show("model.layers.77.mlp.gate.e_score_correction_bias", "最后一层对照")

# 3) iHC 门控：源码 init scale=0.01, base 前4=-log(3)≈-1.0986 后4=0
show("model.layers.0.hc_attn_layer.hc_pre.hc_base", "源码 init: 前4=-log3=-1.0986, 后4=0")
show("model.layers.0.hc_attn_layer.hc_pre.hc_scale", "源码 init=0.01")
show("model.layers.0.hc_mlp_layer.hc_pre.hc_base", "同上")
show("model.layers.0.hc_mlp_layer.hc_pre.hc_scale", "同上")
show("model.layers.40.hc_attn_layer.hc_pre.hc_base", "中间层对照")
show("model.layers.40.hc_attn_layer.hc_pre.hc_scale", "中间层对照")
show("model.layers.77.hc_attn_layer.hc_pre.hc_base", "最后一层对照")
show("model.layers.77.hc_attn_layer.hc_pre.hc_scale", "最后一层对照")

# 4) hc_head：源码 init hc_base 全部 -log3, scale=0.01
show("model.hc_head.hc_head_base", "源码 init: 全部=-log3")
show("model.hc_head.hc_head_scale", "源码 init=0.01")

# 5) indexer k_norm（LayerNorm 参数，ckpt BF16）
show("model.layers.0.self_attn.indexer.k_norm.weight", "LayerNorm gamma")
show("model.layers.0.self_attn.indexer.k_norm.bias", "LayerNorm beta")

# 6) MTP 的 sink/bias 对照（BF16 存储；MTP 无任何 hc 参数，见 headers）
show("model.mtp_layers.0.self_attn.learnable_sink_param", "MTP BF16；主干为 F32")
show("model.mtp_layers.0.mlp.gate.e_score_correction_bias", "MTP BF16")
