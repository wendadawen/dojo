# 验证目标：读取 tencent/Hy4-preview 全部 131 个 safetensors 分片的 JSON 头，
# 建立 (name -> dtype, shape, data_offsets, shard) 完整清单，存 headers.json。
# 用途：层数/模块分布交叉验证、参数量核对、小张量实值下载的 offsets 来源。
import json
import struct
import subprocess
import concurrent.futures as cf
from pathlib import Path

REPO = "https://huggingface.co/tencent/Hy4-preview/resolve/main"
OUT = Path(__file__).parent / "headers.json"


def curl_range(url, start, end):
    r = subprocess.run(["curl", "-sL", "-r", f"{start}-{end}", url], capture_output=True)
    return r.stdout


def read_header(shard):
    url = f"{REPO}/{shard}"
    n = struct.unpack("<Q", curl_range(url, 0, 7))[0]
    raw = curl_range(url, 8, 8 + n - 1)
    header = json.loads(raw.decode())
    return shard, n, header


def main():
    idx = json.load(open(Path(__file__).parent / "model.safetensors.index.json"))
    shards = sorted(set(idx["weight_map"].values()))
    print(f"shards: {len(shards)}")
    all_headers = {}
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        for shard, n, header in ex.map(read_header, shards):
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                all_headers[name] = {
                    "dtype": meta["dtype"],
                    "shape": meta["shape"],
                    "data_offsets": meta["data_offsets"],
                    "shard": shard,
                }
    print(f"tensors: {len(all_headers)}")
    OUT.write_text(json.dumps(all_headers))
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
