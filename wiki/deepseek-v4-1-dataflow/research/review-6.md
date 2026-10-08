# 第 6 轮

源码位置核查通过：276 个节点。有结论的席位 4 个。审查者 4 未计入。这一轮没有事实错误，通过。

## 审查者 1

无

## 审查者 2

无

## 审查者 3

无

## 审查者 4

未计入

## 审查者 5

- 措辞｜MoE.forward 连线｜`shared` 未点开到 `Expert.forward`，路由专家的 `mexp` 有点开｜plan.md；model.py:903｜补上 `shared 点开 → Expert.forward`
- 无事实错误

## 修改

- 没有事实错误。
- 页面核对时，`acc`、`madd`、`off` 的第一行用了源码里没有的名字。第一行改回源码里的那一行。折叠里的公式仍只写这一步自己的累加或加偏移。
- `shared` 仍不点开 Expert.forward。那张图是传入 weights 的路由专家，含 emw。共享专家不传 weights。前提里已经写了这一点。
