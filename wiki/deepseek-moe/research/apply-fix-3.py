# -*- coding: utf-8 -*-
"""DeepSeekMoE 概念页第 3 轮审查问题修复脚本。

每条 old 串断言在全文中恰好命中 1 次，避免误伤。
"""

import io
import sys

BASE = "/Users/wendadawen/code/dojo/wiki/deepseek-moe/"

INDEX = [
    # 问题 8：<head> 去掉无法定位的「K3 的」归因；description 不含裸 ASCII 变量 m
    (
        u"把每个专家分成 m 个小专家（组合更灵活）+ 共享专家隔离（承载通用能力）。K3 的 Stable LatentMoE 继承 shared+routed 组织。",
        u"把每个专家分成更小的多个小专家（组合更灵活）+ 共享专家隔离（承载通用能力）。Stable LatentMoE 继承 shared+routed 组织。",
    ),
    (
        u"隔离路由专家。K3 的 Stable LatentMoE 延续了 shared + routed 的组织方式。",
        u"隔离路由专家。Stable LatentMoE 延续了 shared + routed 的组织方式。",
    ),
    # 问题 1：引言裸 m / 裸 ×m 改为 LaTeX
    (
        u"（每个专家切 m 份、激活数也 ×m，计算量守恒但组合更灵活）",
        u"（每个专家切 $m$ 份、激活数也增至 $m$ 倍，计算量守恒但组合更灵活）",
    ),
    # 问题 2：§2 教学简化裸 mK 改为 LaTeX
    (
        u"组合数假设 router 无约束地选任意 mK 个，实际还受负载均衡影响。",
        u"组合数假设 router 无约束地选任意 $mK$ 个，实际还受负载均衡影响。",
    ),
    # 问题 3：删除「打赢了」的拔高，与来源（comparable）及本节表格一致
    (
        u"DeepSeekMoE 2B 不仅打赢了计算更贵的 GShard 2.9B，还接近了同总参数的稠密模型（Dense×16）的性能——而稠密模型是 MoE 模型的理论上界（每个 token 用全部参数，计算量最大）。这意味着 DeepSeekMoE 几乎榨干了 MoE 架构的潜力。",
        u"DeepSeekMoE 2B 与计算更贵的 GShard 2.9B 性能相当（Pile Loss 均 1.808），还接近了同总参数的稠密模型（Dense$\\times$16）的性能——而稠密模型是 MoE 模型的理论上界（每个 token 用全部参数，计算量最大）。这意味着 DeepSeekMoE 已接近 MoE 架构在模型容量上的上界。",
    ),
    # 问题 6：Table 2 记法 2.83B/0.35B 均为专家参数，非模型总参
    (
        u"<td>GShard 2.9B（2.9B 总参/0.35B 激活，1.5× 专家参数与计算）</td>",
        u"<td>GShard 2.9B（专家参数 2.83B/激活 0.35B，1.5 倍专家参数与计算）</td>",
    ),
    # 问题 4：N3 定位补全到真正含该数值的小节
    (
        u"每专家 0.25× 标准 FFN 即 m=4，激活 1+7=8，总参 2.0B/激活 0.3B）：§4.1.3。",
        u"每专家 0.25 倍标准 FFN 即 m=4，激活 1+7=8，总参 2.0B/激活 0.3B）：§4.1.3、§4.2、§4.5；Table 1、Table 2。",
    ),
    # 问题 5：C8 补注 §4.2（1 共享 + 256 路由、激活 8 的具体数值在 §4.2）
    (
        u"DeepSeek-V3 Technical Report arXiv:2412.19437 §2.1.2。",
        u"DeepSeek-V3 Technical Report arXiv:2412.19437 §2.1.2（sigmoid 亲和度、偏置项、aux-loss-free 均衡）、§4.2（1 共享 + 256 路由、激活 8）。",
    ),
    # 问题 7：补 C5 引用（§2 组合数句），与来源章节双向对应
    (
        u"论文的真实数字更夸张<sup>[N8]</sup>",
        u"论文的真实数字更夸张<sup>[C5, N8]</sup>",
    ),
    # 问题 7：补 F1/F2 引用（§1 通用 top-K MoE 公式句）
    (
        u"的公式 $y(x)=\\sum_{i\\in S_k} g_i\\,E_i(x)$）。",
        u"的公式 $y(x)=\\sum_{i\\in S_k} g_i\\,E_i(x)$）<sup>[F1, F2]</sup>。",
    ),
    # 格式：清除正文中的裸 Unicode ×（统一写「倍」或 KaTeX）
    (
        u"2B 实验：DeepSeekMoE 2B（2.0B 总参/0.3B 激活）与计算多 1.5× 的 GShard 2.9B 性能相当",
        u"2B 实验：DeepSeekMoE 2B（2.0B 总参/0.3B 激活）与计算量多 1.5 倍的 GShard 2.9B 性能相当",
    ),
    (
        u"vs GShard 2.9B（1.5× 专家参数与计算），DeepSeekMoE 仅用对手约 2/3 计算",
        u"vs GShard 2.9B（1.5 倍专家参数与计算），DeepSeekMoE 仅用对手约 2/3 计算",
    ),
    (
        u"N1（DeepSeekMoE 2B vs GShard 2.9B，1.5× 专家参数与计算",
        u"N1（DeepSeekMoE 2B vs GShard 2.9B，1.5 倍专家参数与计算",
    ),
    (
        u"N2（DeepSeekMoE 2B 接近 Dense×16 上界）",
        u"N2（DeepSeekMoE 2B 接近 Dense$\\times$16 上界）",
    ),
    (
        u"每专家 0.25× 即 m=4，激活 2+6=8",
        u"每专家 0.25 倍即 m=4，激活 2+6=8",
    ),
    (
        u"推理约 2.5× 7B 稠密速度",
        u"推理约 2.5 倍 7B 稠密速度",
    ),
    # 格式：清除正文中的裸 Unicode →（改用中文连接或增到）
    (
        u"组合数分别从 4→28、120→44 亿",
        u"组合数分别从 4 增到 28、从 120 增到约 44 亿",
    ),
    (
        u"C2（知识混合成因：专家少→token 涵盖多种知识→专家塞杂）：§1。",
        u"C2（知识混合成因：专家少，token 涵盖多种知识，专家塞杂）：§1。",
    ),
    (
        u"C5（组合数 C(N,K)→C(mN,mK)；16/2/4 例子 120→4,426,165,368）：§3.1 正文。",
        u"C5（组合数从 C(N,K) 增到 C(mN,mK)；16/2/4 例子从 120 增到 4,426,165,368）：§3.1 正文。",
    ),
    (
        u"N8（组合数 120→4,426,165,368）：§3.1 正文。",
        u"N8（组合数从 120 增到 4,426,165,368）：§3.1 正文。",
    ),
    (
        u"E2（细粒度专家分割章，4 专家 top-1、$m=2$ → 8 小专家 top-2）",
        u"E2（细粒度专家分割章，4 专家 top-1、$m=2$ 变 8 小专家 top-2）",
    ),
    (
        u"（如 120 → 44 亿）",
        u"（如从 120 增到约 44 亿）",
    ),
]

OVERVIEW = [
    (
        u"（论文例子：120 → 44 亿）",
        u"（论文例子：从 120 增到约 44 亿）",
    ),
    (
        u"GShard 2.9B（1.5× 专家参数与计算）性能相当",
        u"GShard 2.9B（1.5 倍专家参数与计算）性能相当",
    ),
]


def apply(path, pairs):
    with io.open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    for old, new in pairs:
        n = text.count(old)
        if n != 1:
            raise SystemExit(
                u"ASSERT FAIL [%s] 命中 %d 次：\n%r" % (path, n, old)
            )
        text = text.replace(old, new)
    with io.open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(u"OK %s：%d 处替换" % (path, len(pairs)))


if __name__ == "__main__":
    apply(BASE + "index.html", INDEX)
    apply(BASE + "overview.html", OVERVIEW)
    print("ALL DONE")
