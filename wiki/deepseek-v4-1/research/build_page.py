"""把三段内容注入概念页模板，生成 wiki/deepseek-v4-1/index.html。"""
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NAME = "deepseek-v4-1"
DIR = ROOT / "wiki" / NAME

tpl = io.open(ROOT / ".dojo/templates/concept/index.html", encoding="utf-8").read()
parts = [
    io.open(DIR / "research" / f"_content_{i}.html", encoding="utf-8").read()
    for i in (1, 2, 3)
]
content = "\n".join(p.strip("\n") for p in parts)

CONCEPT = "DeepSeek-V4.1-Flash"
TAGLINE = "把每 token 全局 KV 缓存压到 890 字节的三项叠加"

DESC = (
    "DeepSeek-V4.1-Flash 每 token 只占 890 字节全局 KV 缓存：压缩、跨层共享与 FP4 存储三项叠加的结果。"
    "本文逐项拆开 890 的构成，说明解码器的全局 KV 为何由编码器末层投影、一个 query 能看到哪些位置，"
    "以及 8B 与 16B 激活参数的加总过程。全部数值来自官方配置、参考实现与真实 checkpoint 张量头。"
)

SUMMARY = (
    "每 token 全局 KV 缓存 890 字节 = 主 KV 720（$288$ 字节/条目，压缩比 2 的层折合 144、压缩比 1 的层折合 288）"
    "+ 索引器 K 170；解码器全局 KV 由 $H_{L/2}$ 投影，prefill 只跑编码器半栈（激活 8B），decode 跑满全栈（16B）；"
    "可见集 = 窗口 128 个位置 + 索引器选出的 Top-512 压缩条目。"
)

repl = {
    "【首页摘要】": DESC,
    "【首页可渲染摘要；公式使用 $...$】": SUMMARY,
    "【主题，逗号分隔】": "推理系统,内存与缓存",
    "【内容标签】": "KV cache 压缩",
    "【概念名】": CONCEPT,
    "【简要说明核心作用】": TAGLINE,
}

for k, v in repl.items():
    assert tpl.count(k) >= 1, k
    tpl = tpl.replace(k, v)

marker = "<!-- @content"
i = tpl.find(marker)
assert i >= 0, "未找到 @content 标记"
j = tpl.find("-->", i) + 3

out = tpl[:i] + "\n" + content + "\n" + tpl[j:]
io.open(DIR / "index.html", "w", encoding="utf-8").write(out)

print("已生成", DIR / "index.html")
print("行数:", len(out.split("\n")))
resid = [p for p in ("【", "@component", "@copy-start", "@copy-end") if p in out]
print("残留标记:", resid or "无")
print("@content 残留:", "@content" in out)
