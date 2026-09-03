import torch

p, H, N, D = 2, 2, 4, 3
flat = torch.arange(p * H * N * D, dtype=torch.float32)

nhd = flat.as_strided((N, H, D), (H * D, D, 1))
hnd = flat.as_strided((H, N, D), (N * D, D, 1))

print("物理内存前 12 个元素:", flat[:12].tolist())
print("NHD 视图 token0 (所有头):", nhd[0].tolist())
print("NHD 视图 token1 (所有头):", nhd[1].tolist())
print("HND 视图 head0 (整页 token):", hnd[0].flatten().tolist())
print("HND 视图 head1 (整页 token):", hnd[1].flatten().tolist())
print("NHD stride:", nhd.stride(), " HND stride:", hnd.stride())
