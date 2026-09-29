# 第 1 轮

20 个审查者全部返回。标成事实错误的 6 份里，5 份写的是「未发现事实错误」。真正的事实问题只有一处。

## 事实错误

- inv 节点第三行把「仍是 fp32」解释成「from_pretrained 传 dtype 只转权重」。from_pretrained 会在构造模型前改默认 dtype，普通缓冲区会跟着变。inv_freq 仍是 fp32，是因为初始化时用 torch.float 显式算的。

## 修改

- 第三行改成：初始化时用 torch.float 显式算成 fp32，所以不随 from_pretrained 的 dtype 变成 bf16。
- 其余是措辞（公式里的 q 指哪一步、形状写法不统一、mps 时 autocast 回退 cpu、8 个头各复制的说法）。不为此再开一轮。
