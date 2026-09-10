# -*- coding: utf-8 -*-
"""第 3 轮审查问题修复脚本（causal-mask/index.html）。"""
import io

PATH = '/Users/wendadawen/code/dojo/wiki/causal-mask/index.html'

# (old, new, 编号/说明)
EDITS = [
    # 问题4：为「offset by one position」补 C1 出处标注
    (
        '——"the output embeddings are offset by one position"。',
        '——"the output embeddings are offset by one position"。<sup>[C1]</sup>',
        '问题4-出处标注',
    ),
    # 问题4：把「右移一位」的出处 Figure 1 补进正文，并标注起始符号为工程惯例
    (
        '而是把它右移一位（开头补一个起始符号）得到的序列',
        '而是把它右移一位得到的序列（论文 Figure 1 图内把 decoder 输入标注为 "Outputs (shifted right)"；"开头补一个起始符号"是工程实现惯例）',
        '问题4-Figure1',
    ),
    # 问题2：本章编号引用 -> 章节标题引用
    (
        '第 4 章"一次喂入整条目标序列"喂入的正是',
        '"训练时并行、推理时 KV-cache 隐含"一章的"一次喂入整条目标序列"喂入的正是',
        '问题2-第4章',
    ),
    # 问题2 同类：来源章节内另一处编号引用
    (
        '第 2 章"工程中可用足够大的负数（如 $-10^5$）或 <code>-inf</code> 代替 $-\\infty$"',
        '"因果掩码如何机械地实现"一章的"工程中可用足够大的负数（如 $-10^5$）或 <code>-inf</code> 代替 $-\\infty$"',
        '问题2-第2章',
    ),
    # 问题3：符号表 Q,K,V 的维度按论文原文修正
    (
        '  <li>$Q,K,V\\in\\mathbb{R}^{n\\times d}$：标准注意力的 query、key、value 矩阵（见<a href="../../wiki/standard-attention/index.html">标准注意力</a>），$n$ 为序列长度、$d$ 为模型维度；因果掩码不改变它们。</li>',
        '  <li>$Q,K\\in\\mathbb{R}^{n\\times d_k}$、$V\\in\\mathbb{R}^{n\\times d_v}$：标准注意力的 query、key、value 矩阵（见<a href="../../wiki/standard-attention/index.html">标准注意力</a>），$n$ 为序列长度，$d_k$、$d_v$ 分别为 key、value 的维度；因果掩码不改变它们。</li>',
        '问题3-符号',
    ),
    # 问题1：第 5 章正文交叉引用改到第 3 章标题
    (
        '本章开头的 $3$-token 例子正是这一点的可手算验证',
        '"手算 $3$-token 例子"一章中的 $3$-token 例子正是这一点的可手算验证',
        '问题1-正文',
    ),
    # 问题1：第 5 章「本章问题」解答交叉引用改到第 3 章标题
    (
        '排列对称性被打破。本章的 $3$-token 例子正是可手算的验证',
        '排列对称性被打破。"手算 $3$-token 例子"一章中的 $3$-token 例子正是可手算的验证',
        '问题1-解答',
    ),
    # 问题4：来源章节 C1 补 Figure 1 出处
    (
        'up to and including that position."。</p>',
        'up to and including that position."；Figure 1 图内 decoder 输入标注 "Outputs (shifted right)"（即"output embeddings are offset by one position"的图示）。</p>',
        '问题4-C1出处',
    ),
    # 问题4：辅助解释与类比边界补「起始符号」为工程惯例
    (
        '是工程惯例示例（非论文原文），成立条件是未掩分数在 $\\mathcal{O}(1)$ 量级、远小于该负数示例的绝对值。</p>',
        '是工程惯例示例（非论文原文），成立条件是未掩分数在 $\\mathcal{O}(1)$ 量级、远小于该负数示例的绝对值。正文"开头补一个起始符号"同属工程实现惯例（非论文原文），论文 Figure 1 仅把 decoder 输入标注为 "Outputs (shifted right)"。</p>',
        '问题4-辅助解释',
    ),
]


def main():
    text = io.open(PATH, encoding='utf-8').read()
    for old, new, label in EDITS:
        n = text.count(old)
        assert n == 1, '[%s] old 串命中 %d 次（应为 1）' % (label, n)
        text = text.replace(old, new)
    io.open(PATH, 'w', encoding='utf-8').write(text)
    print('已应用 %d 处修改' % len(EDITS))


if __name__ == '__main__':
    main()
