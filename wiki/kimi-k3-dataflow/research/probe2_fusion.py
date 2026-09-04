# 实测二：多模态融合（_merge_input_ids_with_image_features）。
# 直接以 unbound 方式调用官方 modeling_kimi_k3.py 的方法原文，
# 仅以 SimpleNamespace 提供 config 三个常量（media_placeholder_token_id /
# pad_token_id / ignore_index，取官方 config.json 真值）。
import sys
import types
import torch

sys.path.insert(0, ".")
from _loader import load_official

off = load_official("modeling_kimi_k3.py")

PH = 163605   # media_placeholder_token_id
PAD = 163839  # pad_token_id
IGN = -100    # ignore_index

fake_cfg = types.SimpleNamespace(media_placeholder_token_id=PH,
                                 pad_token_id=PAD, ignore_index=IGN)
fake_self = types.SimpleNamespace(config=fake_cfg)
merge = off.KimiK3ForConditionalGeneration._merge_input_ids_with_image_features

ok = fail = 0
def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name} {detail}")
    else:
        fail += 1
        print(f"  FAIL  {name} {detail}")

E = 7168
def emb(ids):
    # 可辨识的 embedding：文本 token i 的 embedding 全填 i
    t = torch.tensor(ids)
    out = torch.zeros(1, len(ids), E)
    for j, v in enumerate(t.tolist()):
        out[0, j] = float(v + 1)
    return out

print("== 1. 单占位符扩展（1 个占位符 -> N 个视觉 token） ==")
ids = [11, 12, 13, PH, 14, 15]              # 6 个位置，其中 1 个占位符
feats = [torch.randn(5, E) * 0 + torch.arange(5, dtype=torch.float32).unsqueeze(1)]  # 5 个视觉 token，第 k 个填 k
am = torch.ones(1, 6, dtype=torch.long)
labels = torch.tensor([[IGN, 11, 12, 13, 14, 15]])
fe, fam, flab, fpos = merge(fake_self, feats, emb(ids), torch.tensor([ids]), am, labels)
check("输出序列长度 = 原长 - 占位符 + 视觉token", fe.shape == (1, 10, E),
      f"{tuple(fe.shape)} = 6 - 1 + 5")
check("attention_mask 同长", fam.shape == (1, 10))
check("labels 同长", flab.shape == (1, 10))
# 文本落位：位置 0-2 是 token 11/12/13，位置 8-9 是 14/15
check("前 3 文本位 = token 11/12/13",
      torch.allclose(fe[0, 0, 0], torch.tensor(12.)) and
      torch.allclose(fe[0, 1, 0], torch.tensor(13.)) and
      torch.allclose(fe[0, 2, 0], torch.tensor(14.)))
check("后 2 文本位 = token 14/15",
      torch.allclose(fe[0, 8, 0], torch.tensor(15.)) and
      torch.allclose(fe[0, 9, 0], torch.tensor(16.)))
# 视觉落位：位置 3-7 是视觉特征 0..4
vis = [fe[0, 3 + j, 0].item() for j in range(5)]
check("视觉特征按序落在 3..7", vis == [0.0, 1.0, 2.0, 3.0, 4.0], f"{vis}")
check("position_ids 连续 0..9", fpos[0].tolist() == list(range(10)))
print(f"  INFO  labels 变换: {flab[0].tolist()}（占位符原位 labels=13 被视觉段吞掉，视觉段填 ignore_index）")

print("== 2. 两个占位符、两张不同大小的图 ==")
ids2 = [21, PH, 22, PH, 23]                  # 5 个位置，2 个占位符
f2 = [torch.zeros(3, E), torch.zeros(2, E)]
for k in range(3): f2[0][k] = 100.0 + k
for k in range(2): f2[1][k] = 200.0 + k
fe2, fam2, _, fpos2 = merge(fake_self, f2, emb(ids2), torch.tensor([ids2]),
                            torch.ones(1, 5, dtype=torch.long), None)
check("长度 = 5 - 2 + 3 + 2", fe2.shape == (1, 8, E), f"{tuple(fe2.shape)}")
seq = [fe2[0, j, 0].item() for j in range(8)]
# 期望: [22(=token21+1), 100,101,102, 23(=token22+1), 200,201, 24(=token23+1)]
check("交错落位正确", seq == [22.0, 100.0, 101.0, 102.0, 23.0, 200.0, 201.0, 24.0], f"{seq}")
check("position_ids 连续", fpos2[0].tolist() == list(range(8)))

print("== 3. 占位符个数与特征段数不匹配时报错 ==")
try:
    merge(fake_self, [torch.zeros(3, E), torch.zeros(2, E)],
          emb([PH]), torch.tensor([[PH]]),
          torch.ones(1, 1, dtype=torch.long), None)
    check("不匹配报错", False, "未抛出")
except (ValueError, RuntimeError) as e:
    check("不匹配报错", True, f"段数>占位符数: {type(e).__name__}（occupation 表 broadcast 即失败）")
try:
    merge(fake_self, [torch.zeros(3, E)],
          emb([PH, PH]), torch.tensor([[PH, PH]]),
          torch.ones(1, 2, dtype=torch.long), None)
    check("占位符多于段也报错", False, "未抛出")
except (ValueError, AssertionError, RuntimeError) as e:
    check("占位符多于段也报错", True,
          f"{type(e).__name__}: {str(e)[:60]}")

print("== 4. 纯文本（无占位符）不进融合 ==")
# 官方 forward 的守卫：pixel_values 非空才调用 merge；直接以空列表调用会 IndexError
src_fw = open("modeling_kimi_k3.py", encoding="utf-8").read()
check("官方 forward 有守卫",
      "pixel_values is not None and len(" in src_fw and "input_ids.shape[1] != 1" in src_fw,
      "pixel_values 非空且非 decode 单步才调用 _merge...")
try:
    merge(fake_self, [], emb([31, 32, 33]), torch.tensor([[31, 32, 33]]),
          torch.ones(1, 3, dtype=torch.long), None)
    check("空列表直调报错", False, "未抛出")
except IndexError:
    check("空列表直调报错", True, "torch.cat([]) IndexError => 纯文本路径不经过融合函数")
ids4 = [31, 32, 33]
check("序列含占位符才算多模态输入", PH not in ids4)

print("== 5. decode 阶段（input_ids.shape[1]==1 + past_kv）不重融合 ==")
# 官方 forward: pixel_values 与 input_ids 同传且 len(ids)==1 时走 KV 掩码分支，不再扩展序列
src = open("modeling_kimi_k3.py", encoding="utf-8").read()
check("官方含 decode 分支",
      "input_ids.shape[1] == 1" in src and "extended_attention_mask" in src,
      "len(ids)==1 且有 past_key_values 时仅修补 attention_mask，不重跑融合")

print("== 6. 与逐 token 等价的静态描述 ==")
# 一个占位符被 occupation table 记为 N，cumsum 决定新位置 => 序列被拉长而非原地替换
occ = torch.ones(1, 6, dtype=torch.long)
occ[0, 3] = 5
new_pos = torch.cumsum(occ, -1) - 1
print(f"  INFO  occupation cumsum -> 新位置映射: {new_pos[0].tolist()}（占位符位被推开 5 格）")

print(f"\n== 汇总: PASS {ok} / FAIL {fail} ==")
sys.exit(1 if fail else 0)
