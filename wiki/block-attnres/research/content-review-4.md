# 正文审查 第 4 轮

## 事实（A 组：短页、长页开头、第 1-3 章）

位置标错｜第 1 章「残差还有另一种写法」段末｜「把此前所有层的结果先压成了一个向量」不在所引 Generalizing Residuals 内｜该说法出自 2-preliminary.tex Limitations L35「a single compressed state that conflates all earlier layer outputs」｜把该分句移到下一段并改挂 Limitations。

位置标错｜第 1 章「现代模型多用 PreNorm」｜「幅度随层数增长、摊薄」不在所引 Limitations 内｜PreNorm 主导与幅度 $O(L)$、稀释出自 0-abstract.tex L2、1-intro.tex L6｜依据补 0-abstract.tex L1-L8、1-intro.tex Introduction。

## 事实（B 组：第 4-6 章）

位置标错｜第 4 章折叠块「块大小与缩放律的数字」｜模型出处与「哪个 $S$ 对应哪个损失」不在所列范围｜figures/block.tex L83-L108 只有坐标点，x 轴 $S$ 的对应在 L58-L59，图题 L112；「16 个 Transformer 块」出自 5-exp.tex L87-L88（Ablation Study）｜范围改为 figures/block.tex L45-L115，依据补 5-exp.tex Ablation Study。

其余各段通过。

## 复算

无（重算 6 层 2 块例子全 6 步与代码输出、缩放律三数与 1.25 倍、块大小五点、深宽比、三处一般公式非退化代入，均一致；本轮四处改动无新算错或前后不一致）。

## 文风

重复｜长页「权重经过 softmax 归一」句｜「各来源要争同一份概率质量」与「来源之间互相竞争」一句内重复｜删后者。

重复｜长页「现代模型多用 PreNorm」与「这类写法把此前所有层的结果」｜相邻两段都讲「靠后的层放大输出以保持占比」｜删 PreNorm 段该分句，该事实由前一段承担。

口语｜无。离题｜无。AI 味｜无。

## 读者

未解释｜长页第 2 章「每一行加起来为 1」｜「行」只在下方折叠块里定义｜改为「一层给前面各来源的权重加起来为 1」。

未解释｜长页第 5 章折叠块标题「消融」｜首次出现无解释｜补一句「消融指去掉某个设计、看效果变差多少」。

未解释｜长页第 4 章折叠块 $\mathcal{L} = 1.891 \times C^{-0.057}$｜符号 $C$ 未说明｜注明「$C$ 是算力预算」。

## 处理

- 事实 A 两条「位置标错」采纳：「压成一个向量」移到下一段并改挂 `2-preliminary.tex Limitations`；PreNorm 段依据补 `0-abstract.tex L1-L8；1-intro.tex Introduction`。
- 事实 B「依据范围」采纳：第 4 章折叠块依据改为 `5-exp.tex Cross-layer access；5-exp.tex Scaling Behavior；5-exp.tex Ablation Study；figures/block.tex L45-L115`。
- 文风两条「重复」采纳：删「来源之间互相竞争」；删 PreNorm 段「靠后的层只能靠放大自己的输出来保持影响」。
- 读者三条「未解释」采纳：「每一行」改「一层给前面各来源的权重」；折叠块补「消融」释义；$C$ 注明是算力预算。
- 本轮事实 B 与文风审查者首派因接口 5xx 中断，已就同一提示词重派，结论为上。
