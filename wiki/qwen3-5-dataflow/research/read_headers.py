"""并发读取 Qwen/Qwen3.5-397B-A17B 全部 94 个 safetensors 分片的 JSON 头。

验证目标：
  1. 得到全部张量的 name/shape/dtype（不下载权重本体）
  2. 张量总数字节数（用于存档凭证）
  3. 供 verify_structure.py / count_params.py 交叉验证
版本固定：Qwen/Qwen3.5-397B-A17B @ 8472618112abcbd45acbcdc58436aff4233c23f7
"""
import json, struct, subprocess, concurrent.futures, pathlib

SHA = "8472618112abcbd45acbcdc58436aff4233c23f7"
BASE = f"https://huggingface.co/Qwen/Qwen3.5-397B-A17B/resolve/{SHA}"
OUT = pathlib.Path(__file__).parent / "headers.json"

idx = json.load(open(pathlib.Path(__file__).parent / "model.safetensors.index.json"))
shards = sorted(set(idx["weight_map"].values()))
print(f"index 声明分片数: {len(shards)}")

def read_header(shard):
    url = f"{BASE}/{shard}"
    for attempt in range(4):
        try:
            r = subprocess.run(["curl", "-sL", "--max-time", "120", "-r", "0-7", url],
                               capture_output=True, timeout=150)
            n = struct.unpack("<Q", r.stdout[:8])[0]
            r2 = subprocess.run(["curl", "-sL", "--max-time", "180", "-r", f"8-{8+n-1}", url],
                                capture_output=True, timeout=240)
            h = json.loads(r2.stdout.decode())
            return shard, {k: {"dtype": v["dtype"], "shape": v["shape"],
                               "data_offsets": v["data_offsets"]} for k, v in h.items()
                           if k != "__metadata__"}, 8 + n
        except Exception as e:
            if attempt == 3:
                print(f"FAIL {shard}: {e}")
                return shard, {}, 0

headers, total_hdr_bytes = {}, 0
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
    for shard, h, nb in ex.map(read_header, shards):
        headers.update(h)
        total_hdr_bytes += nb

print(f"total tensors: {len(headers)}")
print(f"header bytes read: {total_hdr_bytes}")
json.dump(headers, open(OUT, "w"))
print(f"saved -> {OUT}")

import collections
print("\ndtype 分布:", dict(collections.Counter(v["dtype"] for v in headers.values())))
