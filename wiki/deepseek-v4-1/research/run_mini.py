#!/usr/bin/env python3
"""等比缩小模型端到端实测: prefill/decode 一致性 + 真实张量形状抓取.

验证目标 (对应 official/inference/model.py 整体前向):
  1. 同一权重下, 整段 prefill 与 "prefill + 逐 token decode" 在共同位置的 logits 一致
     (覆盖: window 环形缓冲 / compressor 跨步状态 / index cache / 候选池 / engram cache)
  2. forward hook 抓全部子模块真实输入输出形状 -> ckpt/mini_shapes.json (数据流图依据)
  3. 无 NaN/Inf
  4. DSpark: prefill seed + decode 草稿前向跑通
  5. Vision: encode_image + merge_image_embeddings 跑通
运行: /usr/bin/python3 run_mini.py
输出: ckpt/run_mini.out, ckpt/mini_shapes.json
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from mini_model import init_params, load_model_module, mini_args  # noqa: E402

M = load_model_module()
torch.set_default_dtype(torch.bfloat16)
torch.manual_seed(0)

# engram 需要真实 tokenizer 构建压缩映射
from transformers import AutoTokenizer  # noqa: E402

tokenizer = AutoTokenizer.from_pretrained(str(Path(__file__).parent / "official"))

args = mini_args(M)
model = M.Transformer(args, tokenizer)
init_params(model, seed=0)
model.eval()
print("模型构建完成: n_layers =", args.n_layers, "+ mtp", args.n_mtp_layers)

# ---- 形状抓取 ----
shapes = {}
hooks = []


def mk_hook(name):
    def hook(mod, inp, out):
        def shp(t):
            return list(t.shape) if torch.is_tensor(t) else None

        entry = shapes.setdefault(name, {})
        if inp and inp[0] is not None:
            entry.setdefault("in", shp(inp[0]))
        entry["out"] = shp(out) if not isinstance(out, tuple) else [shp(t) for t in out]

    return hook


for name, mod in model.named_modules():
    if len(list(mod.children())) == 0 or isinstance(mod, (M.Block, M.Attention, M.MoE, M.Engram, M.Compressor, M.Indexer)):
        hooks.append(mod.register_forward_hook(mk_hook(name)))

# ---- 1. prefill vs decode 一致性 ----
T = 16
input_ids = torch.randint(0, 129280, (1, T), generator=torch.Generator().manual_seed(42))
print("输入:", T, "个 token (不含图像 span)")

# 路径 A: 整段 prefill
captured = {}


def head_hook(tag):
    def hook(mod, inp, out):
        captured.setdefault(tag, []).append(inp[0].detach().clone())

    return hook


hk = model.head.register_forward_hook(head_hook("A"))
ids_a, logits_a, hidden_a = model(input_ids, 0)
hk.remove()
h_a = captured["A"][0]
print("路径 A (整段 prefill): head 输入", tuple(h_a.shape), "logits", tuple(logits_a.shape))

# 路径 B: prefill 8 + 逐 token decode 8
model2 = M.Transformer(args, tokenizer)
init_params(model2, seed=0)
model2.eval()
SPLIT = 8
hk = model2.head.register_forward_hook(head_hook("B"))
ids_b, lg, hidden_b = model2(input_ids[:, :SPLIT], 0)
for pos in range(SPLIT, T):
    ids_b, lg, _ = model2(input_ids[:, pos : pos + 1], pos)
hk.remove()
h_b = torch.cat([captured["B"][0][:, -1:]] + captured["B"][1:], dim=1)  # prefill 只取末位置 + 8 次 decode
print("路径 B (prefill 8 + decode 8): head 输入", tuple(h_b.shape))

# 共同位置 hidden 对比: 路径 A 的 [SPLIT-1..T-1] vs 路径 B 全部
la = h_a[:, SPLIT - 1 :].float()
lb = h_b.float()
diff = (la - lb).abs()
rel = diff / la.abs().clamp_min(1e-3)
print(f"hidden 最大绝对差: {diff.max().item():.6e}")
print(f"hidden 最大相对差: {rel.max().item():.6e}")
print(f"hidden 平均绝对差: {diff.mean().item():.6e}")
print("NaN/Inf:", bool(torch.isnan(la).any() or torch.isinf(la).any()))
print("hidden 尺度 (abs mean/max):", f"{la.abs().mean().item():.4f} / {la.abs().max().item():.4f}")
print("注: 该差异已在 debug_consistency3~7 中定位到根因 —— 官方 Indexer 的")
print("    shared_attn.index_k 发布时机 (model.py L537-548 挂在 latent is not None 分支,")
print("    而 L554 无条件读), 使 ratio>1 的 kv_source 层在组未满的解码步读到别的层的 index 键;")
print("    修掉后 bf16 残差降到量化噪声级 (5e-3), fp32 下全部层降到 1e-8 纯累加序差异.")

W = model.head.weight.float()
top_a = (la @ W.T).argmax(-1)
top_b = (lb @ W.T).argmax(-1)
print("逐位置 top-1 token 一致率:", (top_a == top_b).float().mean().item())

for h in hooks:
    h.remove()

Path("ckpt/mini_shapes.json").write_text(json.dumps(shapes, indent=1))
print("形状已存 ckpt/mini_shapes.json, 模块数:", len(shapes))

# ---- 4. DSpark ----
print("\n--- DSpark ---")
# 独立实例, 避免与一致性测试的缓存状态耦合
model3 = M.Transformer(args, tokenizer)
init_params(model3, seed=0)
model3.eval()
ids3, lg3, hid3 = model3(input_ids, 0)  # prefill 16 -> ids3 [B], hid3 [1,16,3*dim]
seed = model3.forward_spec(ids3, hid3, 0)  # 只播种 draft 窗口缓存
print("prefill forward_spec 返回 None:", seed is None)
# decode 语义: main_hidden 必须是"当前位置 1 个位置", 故先走一步主干 decode
ids4, lg4, hid4 = model3(input_ids[:, 15:16], 15)
print("decode 步 main_hidden:", tuple(hid4.shape), "(期望 [1, 1, 3*dim])")
res = model3.forward_spec(ids4, hid4, 15)
assert res is not None
draft_ids, draft_logits, confidence = res
print("draft_ids:", tuple(draft_ids.shape), "draft_logits:", tuple(draft_logits.shape), "confidence:", tuple(confidence.shape))
print("draft_ids[0]:", draft_ids[0].tolist(), "(第 0 个 = 输入 token)")
print("confidence[0]:", [f"{v:.4f}" for v in confidence[0].tolist()])
print("confidence 无 NaN:", bool(~torch.isnan(confidence).any()))

# ---- 5. Vision ----
print("\n--- Vision ---")
# 42x42 像素 -> 3x3 patch 网格 -> 1x1 LLM token
patches = torch.randn(9, 3, 14, 14, dtype=torch.bfloat16)  # 3x3 个 patch
emb = model.encode_image(patches, 3, 3)
print("encode_image(3x3 patch 网格):", tuple(emb.shape), "(期望 [1, 64]: 3x3 unshuffle -> 1 token)")

# 84x42 -> 6x3 patch -> 2x1 LLM token
patches2 = torch.randn(18, 3, 14, 14, dtype=torch.bfloat16)
emb2 = model.encode_image(patches2, 3, 6)
print("encode_image(3x6 patch 网格):", tuple(emb2.shape), "(期望 [2, 64])")
print("vision 输出无 NaN:", bool(~torch.isnan(emb.float()).any()))
