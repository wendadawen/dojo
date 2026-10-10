# 审大纲 第 3 轮

## 读者

无。

## 对照资料

- 资料里没有（位置错）｜适用条件与边界｜「与最容易混淆的扩散 LM 一句取舍」在 Tightness of the NELBO 与 Limitations 内都没有，对比内容在别处｜改：改引 Discussion and Prior Work。
- 资料里没有（依据缺）｜每 token 过两次模型的训练算法｜「训练伪代码」不在依据范围内，伪代码在 algs/short_algs.tex｜改：补引 paper/src/algs/short_algs.tex L49-L67。
- 资料里没有（依据缺）｜逐块生成、块内并行的采样｜「采样伪代码」同因｜改：补引 paper/src/algs/short_algs.tex L69-L89。

## 篇幅与冗余

无。

## 处理

三条全部采纳：训练与采样两节补引伪代码文件，边界节的对比句依据改回 Discussion and Prior Work。到达轮数上限，最后一批改动交仲裁核对。

## 最后一批改动核对

交仲裁核对的最后一批改动（第 3 轮之后、仲裁结论）：
- 训练算法依据补 `paper/src/algs/short_algs.tex L49-L67`；采样依据补 `paper/src/algs/short_algs.tex L69-L89`。
- 边界节依据重新加入 `paper/src/iclr2025_conference.tex Discussion and Prior Work`。
- 第 2 轮改动一并列入：开头依据由 L292-L306 改为 `L292-L301；L354-L377`；训练算法补 `L433`；梯度方差节预算 500→400、措辞收窄、删 `Selecting Noise Schedules to Reduce Training Variance`。

仲裁逐条核对结论：六条改动全部「通过」，理由分别是伪代码确在 short_algs.tex 对应行段、对比句确在 Discussion and Prior Work、L433 确为「不能一次前向」的理由句、L292-L306 内扩散内容确被注释而有效正文在 L354-L377、预算与删项后节内断言仍可推出；未引入「必须改」类新问题，未违背前几轮裁定。总判定：通过。
