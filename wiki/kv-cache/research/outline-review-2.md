# 大纲审查 第 2 轮

## 读者

- 顺序错乱｜长页「缓存按层各存一份，每步追加一个 token」｜「层」「逐层」先用后讲｜开头的总览只交代会讲自回归生成和 query/key/value，上一节也没提模型分层，此前没有节点讲过「层」｜在开头把「模型由多层堆叠、每层各有自己的注意力」作为前置概念链接成页，或先用一句话交代
- 顺序错乱｜短页「把每层算过的 key 和 value 存下来」｜标题和「写什么」里的「每层」在短页内没有讲清｜短页不借长页，本节只声明链接注意力页说明 query/key/value，短页内没有交代「层」是什么｜在短页第 1 节先一句话交代模型由多层堆叠，或把「模型层」作为前置概念链接进短页

## 对照资料

- 资料里没有｜缓存按层各存一份，每步追加一个 token｜「逐层独立计算」的依据不在本节点所列两节内，而在 Attention matrices 节｜hf-cache-explanation.md Attention matrices L56「Attention is calculated independently in each layer of the model, and caching is done on a per-layer basis.」｜依据补「Attention matrices」一节

## 篇幅与冗余

- 无

## 处理

读者：

1. 长页「层」先用后讲：成立。开头总览加上「模型由多层堆叠、每层各有自己的注意力」，作为前置概念交代。
2. 短页「每层」未讲清：成立。短页第 1 节先一句交代模型由多层堆叠，再讲缓存。

对照资料：

3. 「逐层独立计算」依据位置：成立。该节点依据补 `blog/hf-cache-explanation.md Attention matrices`。

篇幅与冗余：无问题。

第 2 轮后重跑机械检查通过，派出第 3 轮。
