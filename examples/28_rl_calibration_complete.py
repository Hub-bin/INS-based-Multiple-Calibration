import sys
import os
import shutil

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gymnasium as gym
from gymnasium import spaces
import gtsam
import matplotlib.pyplot as plt  # Matplotlib 추가

# 외부 모듈은 시뮬레이터와 센서만 사용
from src.utils.road_generator import RoadTrajectoryGenerator
from src.sensors.imu import ImuSensor
from src.simulation.profile import TrajectorySimulator


# ==============================================================================
# 0. Corrected Strapdown Navigator (Internal Definition)
# ==============================================================================
class StrapdownNavigator:
    def __init__(self, start_pose, gravity=9.81):
        self.gravity = gravity
        self.params = gtsam.PreintegrationParams.MakeSharedU(gravity)
        self.params.setAccelerometerCovariance(np.eye(3) * 1e-5)
        self.params.setGyroscopeCovariance(np.eye(3) * 1e-6)
        self.params.setIntegrationCovariance(np.eye(3) * 1e-6)
        self.params.setUse2ndOrderCoriolis(False)
        self.params.setOmegaCoriolis(np.zeros(3))
        self.bias = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
        self.pim = gtsam.PreintegratedImuMeasurements(self.params, self.bias)
        self.curr_pose = start_pose
        self.curr_vel = np.zeros(3)
        self.poses = [start_pose]

    def integrate(self, acc, gyr, dt):
        self.pim.integrateMeasurement(acc, gyr, dt)

    def predict(self):
        state = gtsam.NavState(self.curr_pose, self.curr_vel)
        pred_state = self.pim.predict(state, self.bias)
        self.curr_pose = pred_state.pose()
        self.curr_vel = pred_state.velocity()
        self.pim.resetIntegration()
        self.poses.append(self.curr_pose)
        return self.curr_pose

    def zero_velocity_update(self):
        self.curr_vel = np.zeros(3)


# ==============================================================================
# 1. PPO Agent (Tuned)
# ==============================================================================
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0.0)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.feature_net = nn.Sequential(
            nn.Conv1d(7, 32, 5, 2), nn.ReLU(), nn.Conv1d(32, 64, 3, 2), nn.ReLU(), nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 7, 600)
            flat_size = self.feature_net(dummy).shape[1]

        self.actor_fc = nn.Sequential(nn.Linear(flat_size, 128), nn.ReLU())
        self.actor_mean = nn.Linear(128, action_dim)
        self.actor_act = nn.Tanh()
        self.critic = nn.Sequential(nn.Linear(flat_size, 128), nn.ReLU(), nn.Linear(128, 1))
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 2.0)

        self.apply(init_weights)
        nn.init.uniform_(self.actor_mean.weight, -0.001, 0.001)
        nn.init.constant_(self.actor_mean.bias, 0.0)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        feat = self.feature_net(x)
        mean = self.actor_act(self.actor_mean(self.actor_fc(feat)))
        return mean, self.log_std.exp().expand_as(mean), self.critic(feat)


class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=5e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)
        self.mse = nn.MSELoss()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, std, _ = self.policy(state)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action.cpu().numpy()[0], log_prob.item()

    def update(self, buffer, entropy_coef=0.01):
        states = torch.FloatTensor(np.array([b[0] for b in buffer])).to(self.device)
        actions = torch.FloatTensor(np.array([b[1] for b in buffer])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array([b[2] for b in buffer])).to(self.device)
        rewards = torch.FloatTensor(np.array([b[3] for b in buffer])).to(self.device)

        returns, G = [], 0
        for r in reversed(rewards.cpu().numpy()):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        for _ in range(10):
            mean, std, values = self.policy(states)
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            ratio = torch.exp(log_probs - old_log_probs)
            adv = returns - values.squeeze().detach()

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 0.8, 1.2) * adv

            loss = (
                -torch.min(surr1, surr2).mean()
                + 0.5 * self.mse(values.squeeze(), returns)
                - entropy_coef * entropy.mean()
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        self.scheduler.step()


# ==============================================================================
# 2. Calibration Environment (Bias Only, Fixed Physics)
# ==============================================================================
class CalibrationEnv(gym.Env):
    def __init__(self, road_gen):
        self.road_gen = road_gen
        self.action_space = spaces.Box(-1, 1, (6,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (600, 7), np.float32)
        self.temp_coeffs = {
            "acc_lin": np.array([0.001, 0.001, 0.001]),
            "acc_quad": np.zeros(3),
            "gyr_lin": np.array([1e-5, 1e-5, 1e-5]),
        }
        self.imu = ImuSensor(accel_noise=1e-5, gyro_noise=1e-6)

    def reset(self):
        self.sim = TrajectorySimulator(self.road_gen, 0.1)
        self.traj = self.sim.generate_3d_profile(3)
        self.curr_step = 600
        return self._get_obs(600), {}

    def step(self, action):
        est_bias = {"acc": action[:3] * 0.05, "gyr": action[3:] * 0.005}

        eval_len = 50
        if self.curr_step + eval_len >= len(self.traj):
            return self._get_obs(self.curr_step), 0, True, False, {}

        nav = StrapdownNavigator(self.traj[self.curr_step]["pose"], gravity=9.81)
        nav.curr_vel = self.traj[self.curr_step]["vel_world"]

        vel_err_sum = 0
        for i in range(eval_len):
            idx = self.curr_step + i
            d = self.traj[idx]

            ma, mg, _ = self.imu.measure(d["pose"], d["sf_true"], d["omega_body"], d["temp"])
            dt_t = d["temp"] - 20.0
            ma += self.temp_coeffs["acc_lin"] * dt_t
            mg += self.temp_coeffs["gyr_lin"] * dt_t

            nav.integrate(ma - est_bias["acc"], mg - est_bias["gyr"], 0.1)
            nav.predict()
            vel_err_sum += np.linalg.norm(d["vel_world"] - nav.curr_vel)

        self.curr_step += eval_len
        reward = -(vel_err_sum / eval_len) * 1.0 - np.mean(action**2) * 0.01
        return self._get_obs(self.curr_step), reward, False, False, {}

    def _get_obs(self, idx):
        obs = []
        for i in range(idx - 600, idx):
            d = self.traj[i]
            ma, mg, _ = self.imu.measure(d["pose"], d["sf_true"], d["omega_body"], d["temp"])
            dt_t = d["temp"] - 20.0
            ma += self.temp_coeffs["acc_lin"] * dt_t
            mg += self.temp_coeffs["gyr_lin"] * dt_t
            obs.append(np.concatenate([ma / 9.81, mg, [(d["temp"] - 20) / 30]]))
        return np.array(obs, dtype=np.float32)


# ==============================================================================
# 3. Main Logic
# ==============================================================================
def run_all():
    print(">>> [Integrated] Starting RL Calibration (Adaptive LR + Best Save)...")
    road_gen = RoadTrajectoryGenerator((35.1, 129.0), 5000)
    env = CalibrationEnv(road_gen)
    agent = PPOAgent(state_dim=(600, 7), action_dim=6)

    best_reward = -float("inf")
    save_path = "output_high_end/rl_calibrator.pth"
    if not os.path.exists("output_high_end"):
        os.makedirs("output_high_end")
    if not os.path.exists("output_verification"):
        os.makedirs("output_verification")

    # --- Training ---
    print(">>> Training for 300 Episodes...")
    for ep in range(300):
        s, _ = env.reset()
        done = False
        buffer = []
        ep_r = 0
        while not done:
            a, lp = agent.select_action(s)
            ns, r, term, trunc, _ = env.step(a)
            done = term or trunc
            buffer.append((s, a, lp, r, done))
            s = ns
            ep_r += r

        ent_coef = max(0.001, 0.02 * (0.99**ep))
        agent.update(buffer, entropy_coef=ent_coef)

        if ep_r > best_reward:
            best_reward = ep_r
            torch.save(agent.policy.state_dict(), save_path)
            print(f"  ★ New Best at Ep {ep + 1}: {ep_r:.2f} (Saved)")

        if (ep + 1) % 50 == 0:
            print(f"  Ep {ep + 1}: Reward {ep_r:.2f} (LR: {agent.scheduler.get_last_lr()[0]:.2e})")

    print("\n>>> Loading Best Model for Verification...")
    agent.policy.load_state_dict(torch.load(save_path))
    agent.policy.eval()

    # --- Verification ---
    print("\n>>> Verifying Model...")
    sim = TrajectorySimulator(road_gen, 0.1)
    traj = sim.generate_3d_profile(10)

    nav_raw = StrapdownNavigator(traj[0]["pose"], gravity=9.81)
    nav_rl = StrapdownNavigator(traj[0]["pose"], gravity=9.81)
    nav_raw.curr_vel = nav_rl.curr_vel = traj[0]["vel_world"]
    imu = ImuSensor(accel_noise=1e-5, gyro_noise=1e-6)

    obs_buf = []
    path_gt, path_raw, path_rl = [], [], []
    curr_bias = {"acc": np.zeros(3), "gyr": np.zeros(3)}

    # 학습 환경과 동일한 물리 계수 사용
    temp_coeffs = env.temp_coeffs

    for i, d in enumerate(traj):
        gt_p = d["pose"].translation()
        path_gt.append([gt_p.x(), gt_p.y(), gt_p.z()] if hasattr(gt_p, "x") else gt_p)

        # Measure & Bias
        ma, mg, _ = imu.measure(d["pose"], d["sf_true"], d["omega_body"], d["temp"])
        dt_t = d["temp"] - 20.0
        ma += temp_coeffs["acc_lin"] * dt_t
        mg += temp_coeffs["gyr_lin"] * dt_t

        # ZUPT
        if d["speed"] < 0.05:
            nav_raw.zero_velocity_update()
            nav_rl.zero_velocity_update()

        # Raw Nav
        nav_raw.integrate(ma, mg, 0.1)
        nav_raw.predict()
        pr = nav_raw.poses[-1].translation()
        path_raw.append([pr.x(), pr.y(), pr.z()] if hasattr(pr, "x") else pr)

        # RL Nav
        obs_buf.append(np.concatenate([ma / 9.81, mg, [(d["temp"] - 20) / 30]]))
        if len(obs_buf) > 600:
            obs_buf.pop(0)

        if len(obs_buf) == 600 and i % 10 == 0:
            a, _ = agent.select_action(np.array(obs_buf, dtype=np.float32))
            curr_bias = {"acc": a[:3] * 0.05, "gyr": a[3:] * 0.005}

        nav_rl.integrate(ma - curr_bias["acc"], mg - curr_bias["gyr"], 0.1)
        nav_rl.predict()
        pl = nav_rl.poses[-1].translation()
        path_rl.append([pl.x(), pl.y(), pl.z()] if hasattr(pl, "x") else pl)

    path_gt, path_raw, path_rl = np.array(path_gt), np.array(path_raw), np.array(path_rl)
    err_raw = np.linalg.norm(path_gt - path_raw, axis=1)
    err_rl = np.linalg.norm(path_gt - path_rl, axis=1)

    print(f"\n[Validation Results]")
    print(f"  Mean Error (Raw): {np.mean(err_raw):.2f} m")
    print(f"  Mean Error (RL) : {np.mean(err_rl):.2f} m")

    if np.mean(err_rl) < np.mean(err_raw):
        print("✅ SUCCESS: RL Agent outperforms Raw Navigation!")
    else:
        print("❌ FAIL: RL Agent needs more tuning.")

    # [수정] 시각화 추가
    plt.figure(figsize=(10, 6))
    t_ax = np.arange(len(err_raw)) * 0.1 / 60.0
    plt.plot(t_ax, err_raw, "r--", label="Raw (Temp Drift)")
    plt.plot(t_ax, err_rl, "b-", label="RL Agent")
    plt.title("Navigation Error Comparison")
    plt.xlabel("Time (min)")
    plt.ylabel("Error (m)")
    plt.legend()
    plt.grid(True)
    plt.savefig("output_verification/verification_result.png")
    print("Graph saved to output_verification/verification_result.png")


if __name__ == "__main__":
    run_all()
