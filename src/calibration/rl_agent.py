import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class PolicyNetwork(nn.Module):
    """
    상태(State)를 입력받아 행동(Action)의 평균(mu)과 표준편차(std)를 출력하는 정책 신경망
    - Input: 센서 데이터의 통계적 특징 (Mean, Std 등)
    - Output: Calibration Parameters (Bias 6 + Scale/Misalign 9) -> 총 15개
    """

    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Action Mean (행동의 중심값)
        self.mu_head = nn.Linear(hidden_dim, output_dim)

        # Action Log Std (행동의 범위/탐험 정도, 학습 가능하도록 설정)
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = self.mu_head(x)
        std = torch.exp(self.log_std)  # 항상 양수
        return mu, std


class RLAgent:
    def __init__(self, input_dim=12, action_dim=15, lr=0.002):
        self.policy = PolicyNetwork(input_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        # 안정적인 학습을 위한 감마 (Reward Discount) - 여기선 One-step이라 큰 의미 없으나 관례상 유지
        self.gamma = 0.99

    def get_action(self, state):
        """
        State를 받아 확률적으로 Action을 샘플링 (Exploration)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # (1, input_dim)
        mu, std = self.policy(state_tensor)

        # 정규분포 생성
        dist = torch.distributions.Normal(mu, std)

        # 샘플링 (Action)
        action = dist.sample()

        # Log Probability (나중에 Loss 계산용)
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action.detach().numpy().flatten(), log_prob

    def update(self, log_prob, reward):
        """
        Policy Gradient Update (REINFORCE Algorithm)
        Loss = -log_prob * reward (Maximize Reward)
        """
        loss = -log_prob * reward  # Gradient Ascent

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def decode_action(self, action):
        """15차원 벡터를 물리적 의미(Bias, Scale Matrix)로 변환"""
        # 0~2: Accel Bias
        # 3~5: Gyro Bias
        # 6~14: Scale+Misalign Matrix Elements (flattened 3x3) -> Accel만 적용하거나 둘다 적용

        acc_bias = action[0:3]
        gyr_bias = action[3:6]

        # 단순화를 위해 Accel에 대해서만 Scale/Misalign을 푼다고 가정 (Gyro는 Bias만)
        # 혹은 둘 다 푼다면 Action 차원을 늘려야 함. 여기선 Accel Matrix 9개만 추론.
        T_acc_flat = action[6:15]
        T_acc = T_acc_flat.reshape(3, 3)

        # 초기 학습 불안정을 막기 위해 Identity에 가깝게 조정 (Optional)
        # T_final = I + T_predicted
        T_acc = np.eye(3) + T_acc * 0.1

        return acc_bias, gyr_bias, T_acc
