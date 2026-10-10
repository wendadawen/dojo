# 资料冲突

- sigmoid 损失的偏置符号｜paper/arXiv-2303.15343v4.txt L648-L650（3.2 节公式）把偏置写成 $z_{ij}(-t\,\mathbf{x}_i\cdot\mathbf{y}_j+b)$，$b$ 前为负；同文 L196-L198（Algorithm 1）写 `logits = dot(zimg, ztxt.T) * t + b` 后 `l = -sum(log_sigmoid(labels * logits)) / n`，即 $z_{ij}(t\,\mathbf{x}_i\cdot\mathbf{y}_j+b)$，$b$ 前为正；repo/big_vision 的 `sigmoid_loss` 按第二种实现｜页面按 Algorithm 1 与官方实现写 $(t\,\mathbf{x}\cdot\mathbf{y}+b)$
- 偏置初始化的取值｜paper/arXiv-2303.15343v4.txt L661-L662 与 L1521-L1523 取 $b=-10$；repo/big_vision/big_vision/configs/proj/image_text/siglip_lit_coco.py L91 取 `config.model.bias_init = -2.71`｜页面按论文取 $-10$
