import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Conv1d):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0.0)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.feature_net = nn.Sequential(
            nn.Conv1d(in_channels=7, out_channels=32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        dummy_input = torch.zeros(1, 7, 600)
        with torch.no_grad():
            flatten_size = self.feature_net(dummy_input).shape[1]

        self.actor_fc = nn.Sequential(nn.Linear(flatten_size, 128), nn.ReLU())
        self.actor_mean = nn.Linear(128, action_dim)
        self.actor_act = nn.Tanh()

        self.critic = nn.Sequential(nn.Linear(flatten_size, 128), nn.ReLU(), nn.Linear(128, 1))
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

        # [핵심] 가중치 초기화 적용
        self.apply(init_weights)
        # Actor 출력을 0 근처로 초기화 (Near-Zero Action)
        nn.init.uniform_(self.actor_mean.weight, -0.01, 0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        features = self.feature_net(x)

        x_a = self.actor_fc(features)
        mean = self.actor_mean(x_a)
        mean = self.actor_act(mean)

        std = self.log_std.exp().expand_as(mean)
        value = self.critic(features)
        return mean, std, value


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.mse_loss = nn.MSELoss()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, std, _ = self.policy(state)
            dist = Normal(mean, std)
            action = dist.sample()
            action_log_prob = dist.log_prob(action).sum(dim=-1)
        return action.cpu().numpy()[0], action_log_prob.item()

    def update(self, buffer):
        states = torch.FloatTensor(np.array([b[0] for b in buffer])).to(self.device)
        actions = torch.FloatTensor(np.array([b[1] for b in buffer])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array([b[2] for b in buffer])).to(self.device)
        rewards = torch.FloatTensor(np.array([b[3] for b in buffer])).to(self.device)

        returns = []
        discounted_sum = 0
        for r in reversed(rewards.cpu().numpy()):
            discounted_sum = r + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        returns = torch.FloatTensor(returns).to(self.device)

        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        for _ in range(10):
            mean, std, values = self.policy(states)
            values = values.squeeze()
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            ratio = torch.exp(log_probs - old_log_probs)
            advantage = returns - values.detach()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            loss = (
                -torch.min(surr1, surr2).mean()
                + 0.5 * self.mse_loss(values, returns)
                - 0.01 * entropy.mean()
            )
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
