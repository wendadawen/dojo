"""
验证目标：读取 zai-org/GLM-5.3-Flash 全部 62 个 safetensors 分片的 JSON 头，
拿到真实张量名 / shape / dtype，用于交叉验证源码推导出的结构。
方法：HTTP Range 请求，只取每个分片前 8 字节（头长度，小端 u64）+ JSON 头本体，
不下载权重本体（总计 328GB）。
输出：/tmp/glm53f/headers.json
"""
import json, struct, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

BASE = "https://huggingface.co/zai-org/GLM-5.3-Flash/resolve/main"
N = 62


def fetch(i):
    fn = f"model-{i:05d}-of-{N:05d}.safetensors"
    url = f"{BASE}/{fn}"
    r = subprocess.run(["curl", "-sL", "-r", "0-7", url], capture_output=True)
    if len(r.stdout) < 8:
        return fn, None, f"short read: {len(r.stdout)}"
    n = struct.unpack("<Q", r.stdout[:8])[0]
    r2 = subprocess.run(["curl", "-sL", "-r", f"8-{8 + n - 1}", url], capture_output=True)
    try:
        header = json.loads(r2.stdout.decode())
    except Exception as e:
        return fn, None, f"parse fail (hdrlen={n}, got={len(r2.stdout)}): {e}"
    return fn, header, None


out = {}
errs = []
with ThreadPoolExecutor(max_workers=16) as ex:
    for fn, header, err in ex.map(fetch, range(1, N + 1)):
        if err:
            errs.append((fn, err))
            print("ERR", fn, err, file=sys.stderr)
        else:
            out[fn] = header
            print(f"{fn}: {len(header)} entries", file=sys.stderr)

json.dump(out, open("/tmp/glm53f/headers.json", "w"))
total = sum(len(h) for h in out.values())
print(f"shards ok={len(out)} err={len(errs)} total_header_entries={total}")
