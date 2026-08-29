"""实测 7：视频时间戳逐帧拆分与位置分配（Qwen3.5 特有）。

验证目标（对应源码 modeling_qwen3_5.py get_rope_index L1326-1383 + get_vision_position_ids L1240-1290）：
  7A. video_grid_thw 经 repeat_interleave 按帧数拆分，每帧 (1,h,w)
  7B. 每帧独立走视觉位置生成（T=该帧起始位置），帧间推进 max(h,w)/merge
  7C. 序列位置消耗 = 帧数×(h/2)(w/2)，位置轴推进 = 帧数×max(h,w)/2 —— 两笔账
  7D. 对照：同样内容作为单视频 (t,h,w) 不拆分的区别
"""
import torch

MERGE = 2

def get_vision_position_ids(start, t, h, w, interval=1):
    gt, gh, gw = t, h // MERGE, w // MERGE
    pt = torch.arange(gt) * interval
    ph = torch.arange(gh) + start
    pw = torch.arange(gw) + start
    T, H, W = torch.meshgrid(pt, ph, pw, indexing="ij")
    pos = torch.stack([T, H, W], dim=0).reshape(3, -1)
    pos[0] += start
    return pos

print("=== 7A. 视频按帧拆分（源码 L1327-1329） ===")
video_grid_thw = torch.tensor([[3, 8, 16]])            # 1 个视频：3 帧，grid (8,16)
split = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
split[:, 0] = 1
print(f"  输入 video_grid_thw = {video_grid_thw.tolist()}（1 个 3 帧视频）")
print(f"  拆分后 = {split.tolist()}（3 个单帧）")
print("  → 时间戳 <t1> <vs> 帧1 <ve> <t2> <vs> 帧2 <ve> ... 每帧是独立视觉段")

print()
print("=== 7B/7C. 逐帧位置分配与推进 ===")
current_pos = 5
all_pos = []
for f in range(3):
    vp = get_vision_position_ids(current_pos, 1, 8, 16)
    all_pos.append(vp)
    adv = max(8, 16) // MERGE
    print(f"  帧 {f}: 起始位置 {current_pos}，T 维恒 {current_pos}，H ∈ [{current_pos},{current_pos+3}]，W ∈ [{current_pos},{current_pos+7}]，推进 {adv}")
    current_pos += adv
full = torch.cat(all_pos, dim=1)
print(f"  3 帧共 {full.shape[1]} 个视觉 token，位置轴最终推进到 {current_pos}（从 5 出发推进 {current_pos-5}）")
seq_len_cost = 3 * (8//MERGE) * (16//MERGE)
print(f"  序列位置消耗 = 3 帧 × {8//MERGE}×{16//MERGE} = {seq_len_cost}；位置轴推进 = 3 × {max(8,16)//MERGE} = {3*max(8,16)//MERGE}")
print(f"  → 两笔账：{seq_len_cost} 个 token 进注意力，位置轴只走 {3*max(8,16)//MERGE}")

print()
print("=== 7D. 对照：不拆分的 (3,8,16) 单段视频 ===")
vp = get_vision_position_ids(5, 3, 8, 16)
print(f"  T 维 = {vp[0].unique().tolist()}（帧号 0,1,2 偏移起始 5）——不拆分时 T 才承载帧序")
print(f"  拆分时每帧 T = 该帧起始位置（时间戳本身已编码顺序，帧内 T 退化为常数）")
