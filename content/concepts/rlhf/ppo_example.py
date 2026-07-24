"""
PPO 最小可运行实现 — 对应 content/concepts/rlhf/index.html 里的伪代码

这是一个教学版实现，省略了工程细节（分布式、混合精度、大模型），
但保留了 PPO 的核心结构：
  1. Rollout（采样 + 算经验）
  2. Train（多 epoch 训练 + importance ratio + clip）

运行：python3 content/concepts/rlhf/ppo_example.py
依赖：torch（已安装）

场景：词表 5 个 token，每个 prompt 是 1 个 token，response 也是 1 个 token。
reward 由一个假的 reward_model 给出（手工设定的分数表）。
目标是让 policy 学会生成高 reward 的 token。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from dataclasses import dataclass
from typing import List
import math

# ============================================================
# 配置
# ============================================================
VOCAB_SIZE = 5          # 词表大小（token 0..4）
EMBED_DIM = 16           # embedding 维度
HIDDEN_DIM = 32          # 隐藏层维度
PROMPT_TOKEN = 0         # 所有 prompt 都是 token 0
TARGET_TOKEN = 2         # 我们希望 policy 学会生成这个 token（reward 最高）

# PPO 超参
ROLLOUT_BATCH_SIZE = 8   # 每次（rollout）采多少个 prompt
NUM_EPOCHS = 4           # 每批数据训几个 epoch（PPO 核心：样本复用）
MINI_BATCH_SIZE = 4      # 训练时的 mini-batch 大小
CLIP_EPS = 0.2           # PPO clip 范围
KL_COEF = 0.001          # KL 惩罚系数
GAMMA = 1.0              # 折扣因子（单步场景，无未来）
LAMBDA = 1.0             # GAE 参数
LEARNING_RATE = 1e-2
NUM_EPISODES = 20        # 总共跑几个 episode

# 假的 reward 表：生成 token 2 给 1.0 分，其他给低分
REWARD_TABLE = torch.tensor([-0.5, 0.0, 1.0, 0.1, -0.2])


# ============================================================
# 模型定义
# ============================================================
class TinyLLM(nn.Module):
    """
    极简 LLM：embedding + 一个隐藏层 + 输出 logits
    输入一个 token，输出下一个 token 的 logits（词表大小）
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        """input_ids: (batch,) → logits: (batch, vocab_size)"""
        emb = self.embedding(input_ids)       # (B, embed_dim)
        h = F.relu(self.fc1(emb))             # (B, hidden_dim)
        logits = self.fc2(h)                  # (B, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, input_ids):
        """采样一个 token（贪婪解码，简化用）"""
        logits = self.forward(input_ids)      # (B, vocab_size)
        probs = F.softmax(logits, dim=-1)
        # 用多项分布采样（训练时要探索）
        token = torch.multinomial(probs, num_samples=1)
        return token.squeeze(-1)              # (B,)

    def log_prob(self, input_ids, response):
        """算 log π(response | input_ids)，response 是采样的 token"""
        logits = self.forward(input_ids)       # (B, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)
        # 取出 response 对应的 log prob
        return log_probs.gather(-1, response.unsqueeze(-1)).squeeze(-1)


class ValueModel(nn.Module):
    """极简 value model：和 actor 结构一样，但输出标量"""
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        h = F.relu(self.fc1(emb))
        value = self.fc2(h).squeeze(-1)       # (B,)
        return value


# ============================================================
# 经验数据结构
# ============================================================
@dataclass
class Experience:
    prompt: torch.Tensor      # (B,) prompt token
    response: torch.Tensor    # (B,) response token
    old_log_probs: torch.Tensor   # (B,) 采样时 actor 的 log π
    ref_log_probs: torch.Tensor   # (B,) ref_policy 的 log π
    old_values: torch.Tensor      # (B,) critic 的 value
    reward: torch.Tensor          # (B,) RM 给的 reward
    advantage: torch.Tensor = None
    returns: torch.Tensor = None


# ============================================================
# GAE（Generalized Advantage Estimation）
# ============================================================
def compute_gae(values, rewards, gamma=GAMMA, lambd=LAMBDA):
    """
    单步场景简化版 GAE：advantage = reward - value
    （多步场景要从后往前递推，这里只有一步所以直接算）
    """
    advantages = rewards - values
    returns = advantages + values  # R = A + V
    return advantages, returns


# ============================================================
# Reward Model（假的，用固定分数表）
# ============================================================
def reward_model(prompt, response):
    """假的 RM：根据 response token 查表给分"""
    return REWARD_TABLE[response]


# ============================================================
# PPO 训练器
# ============================================================
class PPOTrainer:
    def __init__(self):
        # actor = 待训练的 policy
        self.actor = TinyLLM(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)
        # ref_policy = 冻结的副本（KL 基准）
        self.ref_policy = deepcopy(self.actor)
        for p in self.ref_policy.parameters():
            p.requires_grad = False
        # critic = value model（待训练）
        self.critic = ValueModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)

    def make_experience(self, prompt_token):
        """阶段 1：采样一个 (prompt, response) 并算经验"""
        prompt = torch.tensor([prompt_token])
        # 1.1 当前 actor 生成 response
        response = self.actor.generate(prompt)
        # 1.2 旧策略的 log prob（detach，不参与梯度）
        old_log_probs = self.actor.log_prob(prompt, response).detach()
        # 1.3 ref_policy 的 log prob（用于 KL）
        with torch.no_grad():
            ref_log_probs = self.ref_policy.log_prob(prompt, response)
        # 1.4 critic 预测 value
        with torch.no_grad():
            old_values = self.critic(prompt)
        # 1.5 RM 打分
        reward = reward_model(prompt, response)
        # 1.6 GAE 算 advantage（单步：A = r - V）
        advantages, returns = compute_gae(old_values, reward)
        return Experience(
            prompt=prompt, response=response,
            old_log_probs=old_log_probs,
            ref_log_probs=ref_log_probs,
            old_values=old_values,
            reward=reward,
            advantage=advantages,
            returns=returns,
        )

    def ppo_train(self, experiences: List[Experience]):
        """阶段 2：PPO 训练（多 epoch + importance ratio + clip）"""
        # 拼成 batch
        prompts = torch.cat([e.prompt for e in experiences])
        responses = torch.cat([e.response for e in experiences])
        old_log_probs = torch.cat([e.old_log_probs for e in experiences])
        ref_log_probs = torch.cat([e.ref_log_probs for e in experiences])
        old_values = torch.cat([e.old_values for e in experiences])
        advantages = torch.cat([e.advantage for e in experiences])
        returns = torch.cat([e.returns for e in experiences])

        # advantage 归一化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        stats = {"policy_loss": 0, "value_loss": 0, "kl": 0, "n_steps": 0}

        for epoch in range(NUM_EPOCHS):
            # 打乱数据
            n = len(experiences)
            perm = torch.randperm(n)
            for i in range(0, n, MINI_BATCH_SIZE):
                idx = perm[i:i+MINI_BATCH_SIZE]
                mb_prompts = prompts[idx]
                mb_responses = responses[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_ref_log_probs = ref_log_probs[idx]
                mb_old_values = old_values[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]

                # ===== Actor 更新 =====
                # 当前 actor 重新算 log prob（θ 已经变了）
                new_log_probs = self.actor.log_prob(mb_prompts, mb_responses)
                # importance ratio = π_new / π_old = exp(log π_new - log π_old)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                # PPO clip loss
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # KL 惩罚（防止 actor 离 ref_policy 太远）
                kl = (new_log_probs - mb_ref_log_probs).mean()
                policy_loss += KL_COEF * kl

                self.actor_optim.zero_grad()
                policy_loss.backward()
                self.actor_optim.step()

                # ===== Critic 更新 =====
                new_values = self.critic(mb_prompts)
                # clipped value loss
                values_clipped = mb_old_values + torch.clamp(
                    new_values - mb_old_values, -CLIP_EPS, CLIP_EPS)
                value_loss = 0.5 * torch.max(
                    (new_values - mb_returns) ** 2,
                    (values_clipped - mb_returns) ** 2
                ).mean()

                self.critic_optim.zero_grad()
                value_loss.backward()
                self.critic_optim.step()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["kl"] += kl.item()
                stats["n_steps"] += 1

        return {k: v / max(stats["n_steps"], 1) for k, v in stats.items() if k != "n_steps"}

    @torch.no_grad()
    def evaluate(self):
        """评估：让 actor 贪婪生成 100 次，看生成 target_token 的概率"""
        prompt = torch.tensor([PROMPT_TOKEN])
        logits = self.actor(prompt)
        probs = F.softmax(logits, dim=-1)
        count = 0
        for _ in range(100):
            token = torch.multinomial(probs, num_samples=1).item()
            if token == TARGET_TOKEN:
                count += 1
        return count / 100, probs.squeeze().tolist()


# ============================================================
# 主训练循环
# ============================================================
def main():
    print("=" * 60)
    print("PPO 最小实现 — 让 policy 学会生成 token", TARGET_TOKEN)
    print("=" * 60)
    print(f"配置：vocab={VOCAB_SIZE}, rollout_batch={ROLLOUT_BATCH_SIZE}, "
          f"epochs={NUM_EPOCHS}, clip={CLIP_EPS}")
    print(f"reward 表：{REWARD_TABLE.tolist()}")
    print(f"目标：让 policy 生成 token {TARGET_TOKEN}（reward={REWARD_TABLE[TARGET_TOKEN].item()}）")
    print()

    trainer = PPOTrainer()

    # 训练前评估
    p_target, probs = trainer.evaluate()
    print(f"[训练前] 生成 token {TARGET_TOKEN} 的概率: {p_target:.2%}")
    print(f"         各 token 概率: {[f'{p:.3f}' for p in probs]}")
    print()

    # ============ 外层循环：每个 episode = 采样 + 多 epoch 训练 ============
    for episode in range(NUM_EPISODES):
        # ---------- 阶段 1：Rollout（采样）----------
        experiences = []
        for _ in range(ROLLOUT_BATCH_SIZE):
            exp = trainer.make_experience(PROMPT_TOKEN)
            experiences.append(exp)

        # 看一下这批 response 里有多少个 target token
        responses = [e.response.item() for e in experiences]
        hit = sum(1 for r in responses if r == TARGET_TOKEN)
        avg_reward = sum(e.reward.item() for e in experiences) / len(experiences)

        # ---------- 阶段 2：PPO 训练（多 epoch）----------
        stats = trainer.ppo_train(experiences)

        # 每 5 个 episode 打印一次
        if (episode + 1) % 5 == 0 or episode == 0:
            print(f"[Episode {episode+1:3d}] "
                  f"hit={hit}/{ROLLOUT_BATCH_SIZE} "
                  f"avg_reward={avg_reward:+.3f} "
                  f"policy_loss={stats['policy_loss']:+.4f} "
                  f"value_loss={stats['value_loss']:.4f} "
                  f"kl={stats['kl']:.4f}")

    print()
    # 训练后评估
    p_target, probs = trainer.evaluate()
    print(f"[训练后] 生成 token {TARGET_TOKEN} 的概率: {p_target:.2%}")
    print(f"         各 token 概率: {[f'{p:.3f}' for p in probs]}")
    print()
    if p_target > 0.5:
        print("✓ 训练成功！policy 学会了生成目标 token")
    else:
        print("✗ 训练未收敛，可以多跑几个 episode 或调大 learning rate")


if __name__ == "__main__":
    main()
