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
import matplotlib.pyplot as plt

from src.utils.road_generator import RoadTrajectoryGenerator
from src.sensors.imu import ImuSensor
from src.simulation.profile import TrajectorySimulator


# ------------------------------------------------------------------------------
# 0. Navigator
# ------------------------------------------------------------------------------
class StrapdownNavigator:
    def __init__(self, start_pose, gravity=9.81):
        self.gravity = gravity
        self.params = gtsam.PreintegrationParams.MakeSharedU(gravity)
        self.params.setAccelerometerCovariance(np.eye(3) * 1e-4)  # 파라미터에 맞게 공분산 조정
        self.params.setGyroscopeCovariance(np.eye(3) * 1e-5)
        self.params.setIntegrationCovariance(np.eye(3) * 1e-5)
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


# ------------------------------------------------------------------------------
# 1. PPO Agent
# ------------------------------------------------------------------------------
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
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.5)
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

            loss = (
                -torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv).mean()
                + 0.5 * self.mse(values.squeeze(), returns)
                - entropy_coef * entropy.mean()
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        self.scheduler.step()


# ------------------------------------------------------------------------------
# 2. Env with High-Weight Parameter Reward & Degraded IMU
# ------------------------------------------------------------------------------
class CalibrationEnv(gym.Env):
    def __init__(self, road_gen, train_duration_min=5):
        self.road_gen = road_gen
        self.train_duration_min = train_duration_min
        self.action_space = spaces.Box(-1, 1, (18,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (600, 7), np.float32)

        # [수정] IMU 성능 저하 (오차 계수 2배 증가)
        self.temp_coeffs = {
            "acc_b_lin": np.array([0.002] * 3),  # 0.001 -> 0.002
            "gyr_b_lin": np.array([2e-5] * 3),  # 1e-5 -> 2e-5
            "acc_s_lin": np.array([0.001] * 3),  # 0.0005 -> 0.001 (0.1%/deg)
            "gyr_s_lin": np.array([0.0002] * 3),  # 0.0001 -> 0.0002
        }
        # [수정] 노이즈 증가 (Low-cost IMU Sim)
        self.imu = ImuSensor(accel_noise=5e-5, gyro_noise=5e-6)

    def reset(self):
        self.sim = TrajectorySimulator(self.road_gen, 0.1)
        self.traj = self.sim.generate_3d_profile(total_duration_min=self.train_duration_min)
        self.curr_step = 600
        return self._get_obs(600), {}

    def step(self, action):
        est = {
            "acc_b": action[0:3] * 0.05,
            "gyr_b": action[3:6] * 0.005,
            "acc_s": action[6:9] * 0.02 + 1.0,
            "gyr_s": action[9:12] * 0.02 + 1.0,
            "acc_h": action[12:15] * 0.0005,
            "gyr_h": action[15:18] * 0.00005,
        }

        eval_len = 50

        if self.curr_step + eval_len >= len(self.traj):
            return self._get_obs(self.curr_step), 0, True, False, {"rmse_b": 0.0, "rmse_s": 0.0}

        nav = StrapdownNavigator(self.traj[self.curr_step]["pose"], gravity=9.81)
        nav.curr_vel = self.traj[self.curr_step]["vel_world"]

        vel_err_sum = 0
        prev_sf = np.zeros(3)

        # Reward용 파라미터 오차 계산
        curr_temp = self.traj[self.curr_step]["temp"]
        dt_t = curr_temp - 20.0

        t_ab = self.temp_coeffs["acc_b_lin"] * dt_t
        t_as = 1.0 + self.temp_coeffs["acc_s_lin"] * dt_t

        rmse_b = np.sqrt(np.mean((t_ab - est["acc_b"]) ** 2))
        rmse_s = np.sqrt(np.mean((t_as - est["acc_s"]) ** 2))

        for i in range(eval_len):
            idx = self.curr_step + i
            d = self.traj[idx]

            ma, mg, _ = self.imu.measure(d["pose"], d["sf_true"], d["omega_body"], d["temp"])
            dt_step = d["temp"] - 20.0

            ta_b = self.temp_coeffs["acc_b_lin"] * dt_step
            ta_s = 1.0 + self.temp_coeffs["acc_s_lin"] * dt_step

            diff_sf = d["sf_true"] - prev_sf
            ta_h = np.array([0.0005] * 3) * np.tanh(diff_sf * 10.0)
            prev_sf = d["sf_true"]

            ma = ma * ta_s + ta_b + ta_h
            mg = mg * (1.0 + self.temp_coeffs["gyr_s_lin"] * dt_step) + (
                self.temp_coeffs["gyr_b_lin"] * dt_step
            )

            corr_acc = (ma - est["acc_b"] - est["acc_h"]) / est["acc_s"]
            corr_gyr = (mg - est["gyr_b"]) / est["gyr_s"]

            nav.integrate(corr_acc, corr_gyr, 0.1)
            nav.predict()
            vel_err_sum += np.linalg.norm(d["vel_world"] - nav.curr_vel)

        self.curr_step += eval_len

        reward_vel = -(vel_err_sum / eval_len) * 1.0
        reward_param = -(rmse_b * 20000.0) - (rmse_s * 500000.0)

        reward = reward_vel + reward_param

        info = {"rmse_b": rmse_b, "rmse_s": rmse_s}

        return self._get_obs(self.curr_step), reward, False, False, info

    def _get_obs(self, idx):
        obs = []
        for i in range(idx - 600, idx):
            d = self.traj[i]
            ma, mg, _ = self.imu.measure(d["pose"], d["sf_true"], d["omega_body"], d["temp"])
            dt_t = d["temp"] - 20.0
            ma = ma * (1.0 + self.temp_coeffs["acc_s_lin"] * dt_t) + (
                self.temp_coeffs["acc_b_lin"] * dt_t
            )
            mg = mg * (1.0 + self.temp_coeffs["gyr_s_lin"] * dt_t) + (
                self.temp_coeffs["gyr_b_lin"] * dt_t
            )
            obs.append(np.concatenate([ma / 9.81, mg, [(d["temp"] - 20) / 30]]))
        return np.array(obs, dtype=np.float32)


# ------------------------------------------------------------------------------
# 3. Main Logic (Extended Plots)
# ------------------------------------------------------------------------------
def run_all():
    print(">>> [Final Tune] RL Calibration (Worse IMU + Damping + Full Plot)...")
    road_gen = RoadTrajectoryGenerator((35.1, 129.0), 5000)

    MAX_EPISODES = 1000
    TRAIN_DATA_LEN = 5

    env = CalibrationEnv(road_gen, train_duration_min=TRAIN_DATA_LEN)
    agent = PPOAgent(state_dim=(600, 7), action_dim=18)

    best_param_error = float("inf")
    save_path = "output_high_end/rl_calibrator_full.pth"
    if not os.path.exists("output_high_end"):
        os.makedirs("output_high_end")
    if not os.path.exists("output_verification"):
        os.makedirs("output_verification")

    print(f">>> Training for {MAX_EPISODES} Episodes...")
    for ep in range(MAX_EPISODES):
        s, _ = env.reset()
        done = False
        buffer = []
        ep_r = 0
        ep_rmse_b = 0
        ep_rmse_s = 0
        steps = 0

        while not done:
            a, lp = agent.select_action(s)
            ns, r, term, trunc, info = env.step(a)
            done = term or trunc
            buffer.append((s, a, lp, r, done))
            s = ns
            ep_r += r

            if "rmse_b" in info:
                ep_rmse_b += info["rmse_b"]
                ep_rmse_s += info["rmse_s"]
                steps += 1

        ent_coef = max(0.001, 0.02 * (0.995**ep))
        agent.update(buffer, entropy_coef=ent_coef)

        if steps > 0:
            avg_rmse = (ep_rmse_b / steps) + (ep_rmse_s / steps) * 10.0

            if avg_rmse < best_param_error:
                best_param_error = avg_rmse
                torch.save(agent.policy.state_dict(), save_path)
                print(f"  ★ New Best Params at Ep {ep + 1}: RMSE {avg_rmse:.5f} (Saved)")

            if (ep + 1) % 50 == 0:
                print(f"  Ep {ep + 1}: Reward {ep_r:.1f} | Param RMSE {avg_rmse:.5f}")

    print("\n>>> Loading Best Model for Verification...")
    if os.path.exists(save_path):
        agent.policy.load_state_dict(torch.load(save_path))
    else:
        print("[Warning] No best model found, using last model.")
    agent.policy.eval()

    # --- Verification ---
    print("\n>>> Verifying Model...")
    sim = TrajectorySimulator(road_gen, 0.1)
    traj = sim.generate_3d_profile(10)

    nav_raw = StrapdownNavigator(traj[0]["pose"], gravity=9.81)
    nav_rl = StrapdownNavigator(traj[0]["pose"], gravity=9.81)
    nav_raw.curr_vel = nav_rl.curr_vel = traj[0]["vel_world"]
    imu = ImuSensor(accel_noise=5e-5, gyro_noise=5e-6)  # Same noise as training

    obs_buf = []
    path_gt, path_raw, path_rl = [], [], []

    # [Log all parameters for plotting]
    # Keys: as (AccScale), ab (AccBias), gs (GyroScale), gb (GyroBias)
    log_data = {
        "true_as_z": [],
        "est_as_z": [],
        "true_ab_z": [],
        "est_ab_z": [],
        "true_gs_z": [],
        "est_gs_z": [],
        "true_gb_z": [],
        "est_gb_z": [],
    }

    curr_params = {
        "acc_b": np.zeros(3),
        "gyr_b": np.zeros(3),
        "acc_s": np.ones(3),
        "gyr_s": np.ones(3),
        "acc_h": np.zeros(3),
        "gyr_h": np.zeros(3),
    }
    temp_coeffs = env.temp_coeffs
    prev_sf = np.zeros(3)

    for i, d in enumerate(traj):
        gt_p = d["pose"].translation()
        path_gt.append([gt_p.x(), gt_p.y(), gt_p.z()] if hasattr(gt_p, "x") else gt_p)

        ma, mg, _ = imu.measure(d["pose"], d["sf_true"], d["omega_body"], d["temp"])

        dt_t = d["temp"] - 20.0
        t_as = 1.0 + temp_coeffs["acc_s_lin"] * dt_t
        t_ab = temp_coeffs["acc_b_lin"] * dt_t
        t_gs = 1.0 + temp_coeffs["gyr_s_lin"] * dt_t
        t_gb = temp_coeffs["gyr_b_lin"] * dt_t

        # Logging True (Z-axis)
        log_data["true_as_z"].append(t_as[2])
        log_data["true_ab_z"].append(t_ab[2])
        log_data["true_gs_z"].append(t_gs[2])
        log_data["true_gb_z"].append(t_gb[2])

        diff_sf = d["sf_true"] - prev_sf
        t_ah = np.array([0.0005] * 3) * np.tanh(diff_sf * 10.0)
        prev_sf = d["sf_true"]

        ma = ma * t_as + t_ab + t_ah
        mg = mg * t_gs + t_gb

        if d["speed"] < 0.05:
            nav_raw.zero_velocity_update()
            nav_rl.zero_velocity_update()

        nav_raw.integrate(ma, mg, 0.1)
        nav_raw.predict()
        pr = nav_raw.poses[-1].translation()
        path_raw.append([pr.x(), pr.y(), pr.z()] if hasattr(pr, "x") else pr)

        obs_buf.append(np.concatenate([ma / 9.81, mg, [(d["temp"] - 20) / 30]]))
        if len(obs_buf) > 600:
            obs_buf.pop(0)

        if len(obs_buf) == 600 and i % 10 == 0:
            a, _ = agent.select_action(np.array(obs_buf, dtype=np.float32))

            # [수정] Damping Factor 적용 (0.5)
            # RL이 너무 과하게 반응하지 않도록 조절
            DAMPING = 0.5

            raw_p = {
                "acc_b": a[0:3] * 0.05 * DAMPING,
                "gyr_b": a[3:6] * 0.005 * DAMPING,
                "acc_s": (a[6:9] * 0.02 * DAMPING) + 1.0,
                "gyr_s": (a[9:12] * 0.02 * DAMPING) + 1.0,
                "acc_h": a[12:15] * 0.0005 * DAMPING,
                "gyr_h": a[15:18] * 0.00005 * DAMPING,
            }
            alpha = 0.1
            for k in curr_params:
                curr_params[k] = (1 - alpha) * curr_params[k] + alpha * raw_p[k]

        # Logging Est
        log_data["est_as_z"].append(curr_params["acc_s"][2])
        log_data["est_ab_z"].append(curr_params["acc_b"][2])
        log_data["est_gs_z"].append(curr_params["gyr_s"][2])
        log_data["est_gb_z"].append(curr_params["gyr_b"][2])

        c_acc = (ma - curr_params["acc_b"] - curr_params["acc_h"]) / curr_params["acc_s"]
        c_gyr = (mg - curr_params["gyr_b"]) / curr_params["gyr_s"]

        nav_rl.integrate(c_acc, c_gyr, 0.1)
        nav_rl.predict()
        pl = nav_rl.poses[-1].translation()
        path_rl.append([pl.x(), pl.y(), pl.z()] if hasattr(pl, "x") else pl)

    path_gt, path_raw, path_rl = np.array(path_gt), np.array(path_raw), np.array(path_rl)
    err_raw = np.linalg.norm(path_gt - path_raw, axis=1)
    err_rl = np.linalg.norm(path_gt - path_rl, axis=1)

    print(f"\n[Validation Results (Full Params + Damping)]")
    print(f"  Mean Error (Raw): {np.mean(err_raw):.2f} m")
    print(f"  Mean Error (RL) : {np.mean(err_rl):.2f} m")

    s_rmse = np.sqrt(
        np.mean((np.array(log_data["true_as_z"]) - np.array(log_data["est_as_z"])) ** 2)
    )
    b_rmse = np.sqrt(
        np.mean((np.array(log_data["true_ab_z"]) - np.array(log_data["est_ab_z"])) ** 2)
    )
    print(f"  > Acc Z Scale RMSE: {s_rmse:.6f}")
    print(f"  > Acc Z Bias RMSE : {b_rmse:.6f}")

    # [수정] 3x2 Grid Plot (Gyro 포함)
    plt.figure(figsize=(18, 12))

    # 1. Trajectory
    plt.subplot(3, 2, 1)
    plt.plot(path_gt[:, 0], path_gt[:, 1], "k", label="GT")
    plt.plot(path_raw[:, 0], path_raw[:, 1], "r--", label="Raw")
    plt.plot(path_rl[:, 0], path_rl[:, 1], "b-", label="RL")
    plt.title("Trajectory")
    plt.legend()
    plt.grid(True)

    # 2. Pos Error
    plt.subplot(3, 2, 2)
    t_ax = np.arange(len(err_raw)) * 0.1 / 60.0
    plt.plot(t_ax, err_raw, "r--", label="Raw")
    plt.plot(t_ax, err_rl, "b-", label="RL")
    plt.title("Position Error")
    plt.legend()
    plt.grid(True)

    # 3. Acc Scale
    plt.subplot(3, 2, 3)
    plt.plot(log_data["true_as_z"], "k--", label="True")
    plt.plot(log_data["est_as_z"], "g-", label="Est")
    plt.title("Acc Scale Z")
    plt.legend()
    plt.grid(True)

    # 4. Acc Bias
    plt.subplot(3, 2, 4)
    plt.plot(log_data["true_ab_z"], "k--", label="True")
    plt.plot(log_data["est_ab_z"], "m-", label="Est")
    plt.title("Acc Bias Z")
    plt.legend()
    plt.grid(True)

    # 5. Gyro Scale
    plt.subplot(3, 2, 5)
    plt.plot(log_data["true_gs_z"], "k--", label="True")
    plt.plot(log_data["est_gs_z"], "g-", label="Est")
    plt.title("Gyro Scale Z")
    plt.legend()
    plt.grid(True)

    # 6. Gyro Bias
    plt.subplot(3, 2, 6)
    plt.plot(log_data["true_gb_z"], "k--", label="True")
    plt.plot(log_data["est_gb_z"], "m-", label="Est")
    plt.title("Gyro Bias Z")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("output_verification/verification_result_full.png")
    print("Graph saved to output_verification/verification_result_full.png")


if __name__ == "__main__":
    run_all()
