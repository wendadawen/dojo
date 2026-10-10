# KV cache 布局

- 描述：KV cache 布局指每一层 key、value 缓存张量的各个维度在显存里的排列顺序，常见的两种记作 NHD 和 HND，区别是序列位置维和头维谁排在前。
- 摘要：同一份 KV cache，NHD 把序列位置排在头之前，每个位置的各个头挨在一起；HND 把头排在序列位置之前，同一个头在整段序列上的向量挨在一起。布局决定一次读相邻数据时拿到的是同一个位置的别的头，还是同一个头的相邻位置。
- 主题：推理系统, 内存与缓存
- 标签：KV cache

## 长页

- 开头｜布局是整组 KV cache 各维的排列顺序，NHD、HND 是其间最常见的两种；它决定读缓存时相邻的是哪一维；链接 KV cache、标准注意力｜200｜blog/flashinfer-kv-layout.rst L9-L20；repo/flashinfer/layout.cuh L28-L48；repo/vllm/v1_kv_cache_layout.py L11-L28
- ### 排在后三维里的两个维度｜整组缓存的逻辑形状是层、块、头、序列位置、每头向量五维，NHD、HND 只交换序列位置维与头维，层、块不动；只有标题，内容在两个小节｜0｜blog/flashinfer-kv-layout.rst L9-L12；repo/vllm/v1_kv_cache_layout.py L11-L28
- #### 两种顺序的形状｜缓存按页分配、每页放固定一段位置；一页内的后三维，NHD 写成位置、头、每头向量，HND 写成头、位置、每头向量；取小数字排一遍元素折叠｜340｜blog/flashinfer-kv-layout.rst L123-L142；repo/vllm/v1_kv_cache_layout.py L23-L28
- #### 相邻数据是哪一维｜一份头向量排完后接什么：NHD 接同位置的下一个头，HND 接同头的下一个位置；给沿序列、沿头步长并折叠｜330｜repo/flashinfer/layout.cuh L35-L48
- ### 为什么会有两种｜NHD 与投影输出排列一致、不必转置；HND 在低精度缓存下对 GPU 实现更友好；两者在 fp16 缓存上无显著差别，默认 NHD｜290｜blog/flashinfer-kv-layout.rst L14-L20
- ### 适用条件与边界｜布局只对多头缓存成立：MLA 把 key、value 合成一个张量、没有头维，就没有 NHD、HND 之分；两种顺序是同样的维度换了先后｜250｜blog/flashinfer-kv-layout.rst L149-L155；blog/flashinfer-kv-layout.rst L9-L12

## 短页

- ### 缓存里各维怎么排｜是什么：累积的 key、value 每层各存一份（见 KV cache）；数按位置、头、每头向量排，NHD 位置在前各头挨着，HND 头在前各位置挨着｜300｜blog/flashinfer-kv-layout.rst L9-L12；blog/flashinfer-kv-layout.rst L123-L129
- ### 读一段缓存先读到什么｜为什么需要：算注意力要同时读查询和整段缓存，一次读进一段相邻数（见标准注意力）；NHD 读到同一位置的各头，HND 读到同一头的一串位置｜300｜repo/flashinfer/layout.cuh L28-L48；blog/flashinfer-kv-layout.rst L14-L16
- ### 选 NHD 还是 HND｜怎么做：两种只是同一批维度换序；选 NHD 因与投影输出一致、免转置，选 HND 因低精度格式下对 GPU 实现更友好；默认 NHD｜300｜blog/flashinfer-kv-layout.rst L14-L20
