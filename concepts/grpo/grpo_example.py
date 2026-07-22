"""
GRPO 实际可运行代码（基于 TRL 库）

依赖安装：
    pip install trl transformers torch datasets

运行：
    python concepts/grpo/grpo_example.py

用一个小模型 + 简单数学题演示 GRPO 训练。
CPU 也能跑（慢），有 GPU 更快。

关键点：
- 用 TRL 的 GRPOTrainer（HuggingFace 官方实现，和 DeepSeekMath 论文对齐）
- reward 是规则验证（RLVR）：答案对=1，错=0
- 组内归一化 advantage（GRPO 核心）由 trainer 自动算
- 没有 critic / value model（GRPO 区别于 PPO 的核心）

看什么：
- rewards/correct_reward/mean：答对率（0~1）
- reward_std：组内 reward 的标准差（=0 说明全对或全错，没有学习信号）
- frac_reward_zero_std：有多少组 reward std=0（GRPO 的"自动跳过"机制）
- kl：当前策略偏离 ref_policy 的程度
- clip_ratio/high_mean、clip_ratio/low_mean：PPO clip 触发的比例
"""

import re
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig


# ============================================================
# 1. 准备数据：简单的加法题
# ============================================================
# prompt 是题目，answer 是正确答案（传给 reward 函数）
# 每条数据会采 num_generations 个回答，组内归一化算 advantage
dataset = Dataset.from_dict({
    "prompt": [
        "1+1=",
        "2+3=",
        "1+2=",
        "2+2=",
        "3+1=",
        "1+3=",
        "3+2=",
        "2+1=",
    ] * 20,  # 重复 20 次凑够训练数据
    "answer": ["2", "5", "3", "4", "4", "4", "5", "3"] * 20,
})


# ============================================================
# 2. 定义 reward 函数（RLVR：答案对错验证）
# ============================================================
def correct_reward(completions, answer, **kwargs):
    """
    检查模型生成的回答里有没有正确答案。

    Args:
        completions: list[str]，每个是模型生成的回答
        answer: list[str]，正确答案（从 dataset 的 "answer" 列来）
        **kwargs: trainer 还会传 prompts, completion_ids 等，用不到就忽略

    Returns:
        list[float]：答对=1.0，答错=0.0
    """
    rewards = []
    for completion, ans in zip(completions, answer):
        # 从生成的文本里提取第一个数字
        match = re.search(r"\d+", completion.strip())
        if match and match.group() == ans:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


# ============================================================
# 3. 配置 GRPO
# ============================================================
# 参数对齐 DeepSeekMath 论文 / TRL 默认值
config = GRPOConfig(
    output_dir="./grpo_output",

    # === GRPO 核心参数 ===
    num_generations=4,           # G：每个 prompt 采 4 个回答（组大小）
    num_iterations=1,            # μ：每批样本训 1 轮（PPO 通常 4，GRPO 常用 1）
    max_completion_length=16,    # 生成长度（加法题很短，16 够用）
    scale_rewards="group",       # 组内归一化（GRPO 核心，TRL 默认）

    # === PPO 继承的参数 ===
    epsilon=0.2,                 # PPO clip 范围 [1-ε, 1+ε]
    beta=0.01,                   # KL 惩罚系数（0 = 不加 KL、不加载 ref_model）

    # === 训练参数 ===
    per_device_train_batch_size=8,
    learning_rate=1e-5,
    temperature=1.0,             # 采样温度（高=探索，低=利用）
    logging_steps=1,
    max_steps=20,                # 跑 20 步看效果（实际训练要几千步）
    save_strategy="no",
    report_to="none",            # 不用 wandb/tensorboard
)


# ============================================================
# 4. 创建 trainer 并训练
# ============================================================
# 模型选择：
#   "sshleifer/tiny-gpt2"  —— 极小（~3MB），CPU 秒跑，但太小不会做数学（reward 一直是 0）
#   "gpt2"                 —— 124M，CPU 能跑（几分钟），偶尔能生成数字
#   "Qwen/Qwen2.5-0.5B-Instruct" —— 0.5B，需要 GPU，能真正学会简单加法
MODEL = "gpt2"

trainer = GRPOTrainer(
    model=MODEL,
    reward_funcs=correct_reward,
    args=config,
    train_dataset=dataset,
)

if __name__ == "__main__":
    print("=" * 60)
    print("开始 GRPO 训练")
    print(f"  模型: {MODEL}")
    print(f"  组大小 G: {config.num_generations}")
    print(f"  clip ε: {config.epsilon}")
    print(f"  KL β: {config.beta}")
    print(f"  reward 归一化: {config.scale_rewards}")
    print("=" * 60)

    trainer.train()

    print("=" * 60)
    print("训练完成")
    print("=" * 60)

    # 测试一下训练后的模型
    test_prompts = ["1+1=", "2+3=", "1+2="]
    print("\n训练后生成示例：")
    for prompt in test_prompts:
        # 用 greedy decoding 看模型学到了什么
        inputs = trainer.processing_class(prompt, return_tensors="pt").to(trainer.model.device)
        output = trainer.model.generate(**inputs, max_new_tokens=16, do_sample=False)
        response = trainer.processing_class.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"  {prompt} -> {response.strip()!r}")
