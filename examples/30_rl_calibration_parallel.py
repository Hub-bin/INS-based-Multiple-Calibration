import sys
import os
import multiprocessing
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import AsyncVectorEnv
import gtsam
import matplotlib.pyplot as plt

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.road_generator import RoadTrajectoryGenerator
from src.sensors.imu import ImuSensor
from src.simulation.profile import TrajectorySimulator


# ==============================================================================
# 1. Fast Network (MLP) - Tanh Essential
# ==============================================================================
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0.0)


class FastActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(FastActorCritic, self).__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),  # 용량 증대
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 2.0)

        # [중요] Tanh 활성화 함수 (Action Bound 준수)
        self.act = nn.Tanh()
        self.apply(init_weights)
        nn.init.uniform_(self.actor.weight, -0.001, 0.001)
        nn.init.constant_(self.actor.bias, 0.0)

    def forward(self, x):
        # x: [Batch, Window, Sensors]
        mean = x.mean(dim=1)
        std = x.std(dim=1) + 1e-6
        # Drift: (End - Start) -> 히스테리시스 경향성 파악용
        drift = x[:, -1, :] - x[:, 0, :]

        # Input Dim = 7*3 = 21
        feat = torch.cat([mean, std, drift], dim=1)
        h = self.shared_net(feat)
        return self.act(self.actor(h)), self.log_std.exp(), self.critic(h)


# ==============================================================================
# 2. Environment (Long Horizon & Full Physics)
# ==============================================================================
class AdvancedCalibrationEnv(gym.Env):
    def __init__(self, seed_offset=0, train_duration_min=10):
        self.road_gen = RoadTrajectoryGenerator((35.1, 129.0), 5000)
        self.action_space = spaces.Box(-1, 1, (18,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (600, 7), np.float32)
        self.train_duration_min = train_duration_min

        # [Physics] Hysteresis 포함, Large Drift
        self.physics = {
            "coeffs": {
                "acc_b_lin": np.array([0.005] * 3),  # 1도당 5mg
                "gyr_b_lin": np.array([5e-5] * 3),
                "acc_s_lin": np.array([0.002] * 3),  # 1도당 0.2%
                "gyr_s_lin": np.array([0.0005] * 3),
                "acc_h_tanh": 0.005,  # 히스테리시스 크기 5mg
                "gyr_h_tanh": 0.0005,
            }
        }
        self.imu = ImuSensor(accel_noise=1e-4, gyro_noise=1e-5)
        self.seed_offset = seed_offset

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed + self.seed_offset)
        self.sim = TrajectorySimulator(self.road_gen, 0.1)
        # [핵심] 10분 데이터 생성 (장기 패턴 학습)
        self.traj = self.sim.generate_3d_profile(total_duration_min=self.train_duration_min)
        self.curr_step = 600
        self.prev_sf = np.zeros(3)
        self.prev_wb = np.zeros(3)
        return self._get_obs(600), {}

    def step(self, action):
        # [Action Range] 충분히 넓게 설정 (Saturation 방지)
        est = {
            "acc_b": action[0:3] * 0.2,  # +/- 200mg
            "gyr_b": action[3:6] * 0.01,
            "acc_s": action[6:9] * 0.1 + 1.0,  # +/- 10%
            "gyr_s": action[9:12] * 0.02 + 1.0,
            "acc_h": action[12:15] * 0.01,  # +/- 10mg
            "gyr_h": action[15:18] * 0.001,
        }

        eval_len = 50
        if self.curr_step + eval_len >= len(self.traj):
            return self._get_obs(self.curr_step), 0.0, True, False, {"rmse_b": 0.0}

        mse_sum = 0.0
        coeffs = self.physics["coeffs"]

        for i in range(eval_len):
            d = self.traj[self.curr_step + i]
            dt_t = d["temp"] - 20.0

            # True Params
            t_as = 1.0 + coeffs["acc_s_lin"] * dt_t
            t_ab = coeffs["acc_b_lin"] * dt_t
            t_gs = 1.0 + coeffs["gyr_s_lin"] * dt_t
            t_gb = coeffs["gyr_b_lin"] * dt_t

            # True Hysteresis (Stateful)
            diff_sf = d["sf_true"] - self.prev_sf
            diff_wb = d["omega_body"] - self.prev_wb
            t_ah = coeffs["acc_h_tanh"] * np.tanh(diff_sf * 50.0)
            t_gh = coeffs["gyr_h_tanh"] * np.tanh(diff_wb * 50.0)

            self.prev_sf = d["sf_true"]
            self.prev_wb = d["omega_body"]

            # Reward: Parameter Matching
            l_ab = np.mean((t_ab - est["acc_b"]) ** 2) * 1500.0
            l_as = np.mean((t_as - est["acc_s"]) ** 2) * 5000.0
            l_ah = np.mean((t_ah - est["acc_h"]) ** 2) * 10000.0

            l_gb = np.mean((t_gb - est["gyr_b"]) ** 2) * 500000.0
            l_gs = np.mean((t_gs - est["gyr_s"]) ** 2) * 150000.0
            l_gh = np.mean((t_gh - est["gyr_h"]) ** 2) * 1000000.0

            mse_sum += l_ab + l_as + l_ah + l_gb + l_gs + l_gh

        self.curr_step += eval_len

        # Info for Logging
        rmse_b = np.sqrt(np.mean((t_ab - est["acc_b"]) ** 2))

        return (
            self._get_obs(self.curr_step),
            -(mse_sum / eval_len),
            False,
            False,
            {"rmse_b": rmse_b},
        )

    def _get_obs(self, idx):
        start = idx - 600
        obs = []
        for i in range(start, idx):
            d = self.traj[i]
            ma, mg, _ = self.imu.measure(d["pose"], d["sf_true"], d["omega_body"], d["temp"])
            # Obs에는 Hysteresis 제외 (노이즈로 취급하여 AI가 패턴을 찾도록 유도)
            dt_t = d["temp"] - 20.0
            t_as = 1.0 + self.physics["coeffs"]["acc_s_lin"] * dt_t
            t_ab = self.physics["coeffs"]["acc_b_lin"] * dt_t
            ma = ma * t_as + t_ab

            t_gs = 1.0 + self.physics["coeffs"]["gyr_s_lin"] * dt_t
            t_gb = self.physics["coeffs"]["gyr_b_lin"] * dt_t
            mg = mg * t_gs + t_gb

            obs.append(np.concatenate([ma / 9.81, mg, [(d["temp"] - 20) / 30]]))
        return np.array(obs, dtype=np.float32)


# ------------------------------------------------------------------------------
# 3. PPO Agent
# ------------------------------------------------------------------------------
class PPOAgent:
    def __init__(self, input_dim, action_dim, lr=5e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = FastActorCritic(input_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.mse = nn.MSELoss()

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            mean, std, _ = self.policy(state)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action.cpu().numpy(), log_prob.cpu().numpy()

    def update(self, batch_s, batch_a, batch_r, batch_lp):
        s = torch.FloatTensor(np.concatenate(batch_s)).to(self.device)
        a = torch.FloatTensor(np.concatenate(batch_a)).to(self.device)
        r = torch.FloatTensor(np.concatenate(batch_r)).to(self.device).flatten()
        lp = torch.FloatTensor(np.concatenate(batch_lp)).to(self.device).flatten()

        for _ in range(10):
            mean, std, values = self.policy(s)
            dist = Normal(mean, std)
            log_probs = dist.log_prob(a).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            ratio = torch.exp(log_probs - lp)
            loss = (
                -torch.min(ratio * r, torch.clamp(ratio, 0.8, 1.2) * r).mean()
                - 0.01 * entropy.mean()
            )
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()


# ------------------------------------------------------------------------------
# 4. Runner
# ------------------------------------------------------------------------------
def make_env(idx, duration):
    def _init():
        return AdvancedCalibrationEnv(seed_offset=idx, train_duration_min=duration)

    return _init


def run_parallel():
    NUM_ENVS = 8
    # [핵심 수정] 끝장 학습 설정
    MAX_EPISODES = 3000
    TRAIN_DATA_LEN = 10

    print(
        f">>> [Parallel] Launching {NUM_ENVS} Environments (Duration {TRAIN_DATA_LEN}m, Ep {MAX_EPISODES})..."
    )
    envs = AsyncVectorEnv([make_env(i, TRAIN_DATA_LEN) for i in range(NUM_ENVS)])
    agent = PPOAgent(input_dim=21, action_dim=18)

    states, _ = envs.reset()
    best_loss = float("inf")
    if not os.path.exists("output_high_end"):
        os.makedirs("output_high_end")

    for update in range(MAX_EPISODES):
        batch_s, batch_a, batch_r, batch_lp = [], [], [], []

        # Data Collection
        for _ in range(50):
            a, lp = agent.select_action(states)
            ns, r, _, _, infos = envs.step(a)
            batch_s.append(states)
            batch_a.append(a)
            batch_r.append(r)
            batch_lp.append(lp)
            states = ns

        agent.update(batch_s, batch_a, batch_r, batch_lp)
        avg_r = np.mean(batch_r)

        # Save Best
        if -avg_r < best_loss:
            best_loss = -avg_r
            torch.save(agent.policy.state_dict(), "output_high_end/rl_parallel.pth")
            print(f"Update {update + 1} | ★ New Best Loss: {-avg_r:.4f}")

        if (update + 1) % 100 == 0:
            print(f"Update {update + 1} | Loss: {-avg_r:.4f}")

    print(">>> Training Complete.")
    envs.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_parallel()
