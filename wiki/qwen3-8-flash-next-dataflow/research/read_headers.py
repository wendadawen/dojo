"""按固定 sha 用 HTTP Range 读取所有 safetensors 分片的 JSON 头，落盘 headers.json。

不下载权重本体：safetensors 前 8 字节是头长度（小端 u64），其后为 JSON 头。
"""
import json, struct, subprocess, concurrent.futures as cf

REPO = "Qwen/Qwen3.8-Flash-Next"
SHA = "f5d08274bafd880402bd16f5e3e6c514136ec06c"

files = sorted({v for v in json.load(open("/tmp/qwen38fn/model.safetensors.index.json"))["weight_map"].values()})
print("shards:", len(files))


def read_header(fn):
    url = f"https://huggingface.co/{REPO}/resolve/{SHA}/{fn}"
    r = subprocess.run(["curl", "-sL", "-r", "0-7", url], capture_output=True)
    n = struct.unpack("<Q", r.stdout[:8])[0]
    r2 = subprocess.run(["curl", "-sL", "-r", f"8-{8 + n - 1}", url], capture_output=True)
    return fn, json.loads(r2.stdout.decode()), n


out, total_bytes = {}, 0
with cf.ThreadPoolExecutor(16) as ex:
    for fn, hdr, n in ex.map(read_header, files):
        total_bytes += n
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            out[k] = {"dtype": v["dtype"], "shape": v["shape"], "shard": fn}
        print(f"{fn}: {len(hdr)} entries, header {n} B", flush=True)

json.dump(out, open("/tmp/qwen38fn/headers.json", "w"))
print("total tensors:", len(out), "| header bytes read:", total_bytes)
