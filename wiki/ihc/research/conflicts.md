# 资料冲突

- 动态参数的激活函数｜paper/src-v3/iclr2025_conference.tex Dynamic Hyper-Connections 三组参数都用 tanh；paper/dsv4-2606.19348v1.txt L419-L424 的输入、输出映射用 Sigmoid，输出再乘 2，repo/hy4-transformers-modeling.py L603-L641 同为 Sigmoid｜页面分开写：HC 的原始形式写 tanh，DeepSeek-V4 的形式与 Hy4 实现写 Sigmoid，各自注明
- 预归一化等价的初始化｜paper/src-v3/iclr2025_conference.tex Initialization 的初始化针对 tanh 形式（A 的初值为 $\mathbf{e}_{k \bmod n}$，C 的初值为 1）；Sigmoid 形式的资料里没有同样的说法｜页面只把「起步等价预归一化」归给 HC 的原始形式
- n=2 的显存增量｜paper/src-v3/app.tex Memory Footprint 的文字写 n=2 时「less than 15%」，同一节实测表写 +15.7%｜页面采用表中的 15.7%
- OLMo-1B-DHC×4 的 V3 验证损失｜paper/hc-2409.19606v3.txt Table 1 写 2.514，Table 2 写 2.515｜页面采用 Table 1 的 2.514
- 训练显存增量的说法｜paper/src-v3/app.tex Memory Footprint 的文字写训练时激活显存「minor increase」，同一节实测表（hc txt Table 9）为 OLMo-1B n=4 +26.1%、OLMo-7B +28.28%｜页面只写表中数字，不用「明显」「轻微」这类评价
- iHC 的名称｜blog/hy4-model-card.md 写 identity Hyper-Connections，repo/hy4-transformers-modeling.py L605 类文档串写 Independent Hyper-Connection｜页面采用模型卡的名字；「流间混合等价于恒等」依据代码里没有 B 这一项，不依赖名字
