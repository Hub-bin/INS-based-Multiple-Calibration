import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = self.mu_head(x)
        std = torch.exp(self.log_std)
        return mu, std


class RLAgent:
    def __init__(self, input_dim, action_dim, lr=0.002):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.policy = PolicyNetwork(input_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def get_action(self, state, deterministic=False):
        """
        :param deterministic: True면 탐험 없이 평균값(Mean)만 반환 (Test용)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        mu, std = self.policy(state_tensor)

        if deterministic:
            action = mu
            log_prob = torch.zeros_like(action)  # Not used in test
        else:
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        return action.detach().numpy().flatten(), log_prob

    def update(self, log_prob, reward):
        loss = -log_prob * reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
