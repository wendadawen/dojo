#!/usr/bin/env python3
"""HTTP Range 读取 DeepSeek-V4.1-Flash 全部 48 个 safetensors 分片的 JSON 头。

验证目标: 不下载权重本体, 拿到全部张量的 dtype/shape/data_offsets,
用于交叉验证源码推导的模块分布与参数量核算。
对应: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
输出: ckpt/headers.json (张量名 -> {dtype, shape, shard, data_offsets})
"""
import json
import struct
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main"
N_SHARDS = 48
OUT = Path(__file__).parent / "ckpt" / "headers.json"


def curl_range(url: str, start: int, end: int) -> bytes:
    r = subprocess.run(
        ["curl", "-sL", "--fail", "-r", f"{start}-{end}", url],
        capture_output=True, check=True, timeout=300,
    )
    return r.stdout


def read_one(shard: str) -> tuple[str, dict]:
    url = f"{BASE}/{shard}"
    head = curl_range(url, 0, 7)
    n = struct.unpack("<Q", head[:8])[0]
    raw = curl_range(url, 8, 8 + n - 1)
    return shard, json.loads(raw.decode())


def main() -> None:
    shards = [f"model-{i:05d}-of-{N_SHARDS:05d}.safetensors" for i in range(1, N_SHARDS + 1)]
    out = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        for shard, header in ex.map(read_one, shards):
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                out[name] = {
                    "dtype": meta["dtype"],
                    "shape": meta["shape"],
                    "shard": shard,
                    "data_offsets": meta["data_offsets"],
                }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out))
    print(f"tensors: {len(out)}")
    dtypes = {}
    for m in out.values():
        dtypes[m["dtype"]] = dtypes.get(m["dtype"], 0) + 1
    print("dtypes:", dict(sorted(dtypes.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
