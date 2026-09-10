"""补齐正文中的公式编号引用；删除本页未使用的 F8。"""
import io

FS = "{0:c}".format(92)


def patch(path, pairs, drop_lines=()):
    s = io.open(path, encoding="utf-8").read()
    for old, new in pairs:
        n = s.count(old)
        assert n == 1, (path, n, old[:70])
        s = s.replace(old, new)
    if drop_lines:
        lines = s.split("\n")
        before = len(lines)
        lines = [l for l in lines if not any(l.startswith(p) for p in drop_lines)]
        s = "\n".join(lines)
        print(f"  {path} 删除行数: {before - len(lines)}")
    io.open(path, "w", encoding="utf-8").write(s)
    print("已更新", path, "共", len(pairs), "处替换")


patch("_content_1.html", [
    (
        "投影权重按层各自不同<sup>[C6]</sup>：</p>",
        "投影权重按层各自不同<sup>[C6, F7]</sup>：</p>",
    ),
])

patch("_content_2.html", [
    (
        "<li><b>块级</b>：把位置切成大小 8 的块，块分取块内最大的索引分。建池层为每个 query 选出 2048 个块。</li>",
        "<li><b>块级</b>：把位置切成大小 8 的块，块分取块内最大的索引分<sup>[F4]</sup>。建池层为每个 query 选出 2048 个块。</li>",
    ),
    (
        "<p>第二级的搜索范围从「全部可达条目」缩到 16384 个候选，缩小了约两个数量级。两级用的是同一套索引分，区别只在聚合粒度。</p>",
        "<p>第二级的搜索范围从「全部可达条目」缩到 16384 个候选，缩小了约两个数量级。两级用的是同一套索引分——每个 query 对每条压缩条目的打分 $"
        + FS
        + "sum_h w_{q,h}"
        + FS
        + "mathrm{ReLU}(q_h"
        + FS
        + "cdot k_j)$<sup>[F2]</sup>，区别只在聚合粒度。</p>",
    ),
])

patch(
    "_content_3.html",
    [
        (
            "选择时加上一个可学习的修正偏置（见 <a href=\"../deepseek-moe/index.html\">DeepSeek MoE</a> 与 <a href=\"../aux-loss-free-routing/index.html\">无辅助损失负载均衡</a>）<sup>[C8]</sup>：</p>",
            "选择时加上一个可学习的修正偏置（见 <a href=\"../deepseek-moe/index.html\">DeepSeek MoE</a> 与 <a href=\"../aux-loss-free-routing/index.html\">无辅助损失负载均衡</a>）<sup>[C8, F5]</sup>：</p>",
        ),
        (
            "<p>超连接把残差流按 4 份并行携带，用三套系数读写（见 <a href=\"../hyper-connections/index.html\">超连接</a>）。它的参数量来自一个从隐状态到系数的小投影：",
            "<p>超连接把残差流按 4 份并行携带，用三套系数（读前、写后、以及 4 份之间的混合）读写<sup>[F6]</sup>，其中混合矩阵经 20 轮行列交替归一化逼近双随机<sup>[C9]</sup>（见 <a href=\"../hyper-connections/index.html\">超连接</a>）。它的参数量来自一个从隐状态到系数的小投影：",
        ),
    ],
    drop_lines=("<p>[F8]",),
)
