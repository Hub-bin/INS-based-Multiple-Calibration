import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gtsam

from src.dynamics.ground import GroundVehicle
from src.sensors.imu import ImuSensor
from src.calibration.sysid_corrector import SysIdCalibrator


# --- Internal RLAgent Definition (Customized for Init) ---
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))

        # [핵심 수정] 초기 출력을 양수(0.5)로 설정하여 "일단 켜는" 성향 부여
        # Bias=0(OFF) 근처에서 시작하면 탐험하다가 로컬 미니멈에 빠질 수 있음
        nn.init.constant_(self.mu_head.bias, 1.0)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = self.mu_head(x)
        std = torch.exp(self.log_std)
        return mu, std


class RLAgentCustom:
    def __init__(self, input_dim, action_dim, lr=0.002):
        self.policy = PolicyNetwork(input_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def get_action(self, state, deterministic=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        mu, std = self.policy(state_tensor)

        if deterministic:
            action = mu
            log_prob = torch.zeros_like(action)
        else:
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        return action.detach().numpy().flatten(), log_prob, mu.detach().numpy().flatten()

    def update(self, log_prob, reward):
        loss = -log_prob * reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# --- Simulation Logic ---
def generate_temp_scenario(dt, duration, mode="Stable"):
    steps = int(duration / dt)
    vehicle = GroundVehicle()
    true_data = []
    temps = []

    for i in range(steps):
        t = i * dt
        if mode == "Stable":
            current_temp = 20.0 + np.random.normal(0, 0.01)
        elif mode == "Rising":
            current_temp = 20.0 + (40.0 * t / duration)
        temps.append(current_temp)
        vehicle.update(dt, 0.0, 0.0)

        rot = vehicle.current_pose.rotation()
        g_body_np = rot.unrotate(gtsam.Point3(0, 0, -9.81))
        if not isinstance(g_body_np, np.ndarray):
            g_body_np = np.array([g_body_np.x(), g_body_np.y(), g_body_np.z()])
        sf_body = -g_body_np
        true_data.append((sf_body, np.zeros(3)))

    return vehicle, true_data, temps


def get_normalized_state(temps):
    temp_arr = np.array(temps)
    # Norm: Mean(0~1), Std(0~1)
    return np.array([np.mean(temp_arr) / 100.0, np.std(temp_arr) / 12.0])


def calculate_mse(sysid_result, true_data, raw_data, temps):
    mse_sum = 0.0
    count = len(raw_data)
    est_b = sysid_result["acc_b"]
    est_k1 = sysid_result["acc_k1"]
    est_k2 = sysid_result["acc_k2"]
    ref_temp = 20.0

    for i in range(count):
        raw_acc = raw_data[i][0]
        true_acc = true_data[i][0]
        dt = temps[i] - ref_temp
        correction = est_b + (est_k1 * dt) + (est_k2 * (dt**2))
        est_true_acc = raw_acc - correction
        mse_sum += (true_acc[0] - est_true_acc[0]) ** 2

    return mse_sum / count


def main():
    print("=== RL: Temperature Coefficient Learnability (Initialized Policy) ===")

    dt = 0.1
    duration = 5.0
    episodes = 5000  # 5000회로 증가

    accel_noise = 0.002
    sensor_variance = accel_noise**2

    imu = ImuSensor(
        accel_bias=[0.5, 0.0, 0.0],
        accel_temp_coeff_linear=0.05,
        accel_temp_coeff_nonlinear=0.0005,
        accel_noise=accel_noise,
        gyro_noise=0.0001,
    )

    # Custom Agent 사용
    agent = RLAgentCustom(input_dim=2, action_dim=2, lr=0.001)
    sysid = SysIdCalibrator()
    reward_history = []

    print("Training RL Agent...")

    baseline_mask = np.zeros(18)
    baseline_mask[9:12] = 1.0

    for ep in range(episodes):
        mode = np.random.choice(["Stable", "Rising"])
        _, true_data, temps = generate_temp_scenario(dt, duration, mode)

        raw_meas = []
        for i, (sf, om) in enumerate(true_data):
            m = imu.measure(gtsam.Pose3(), sf, om, temperature=temps[i])
            raw_meas.append((m[0], m[1]))

        state = get_normalized_state(temps)

        # Action
        action, log_prob, _ = agent.get_action(state, deterministic=False)
        selected = (action > 0.0).astype(float)

        acc_mask = np.zeros(18)
        if selected[0]:
            acc_mask[9] = 1.0
        if selected[1]:
            acc_mask[12] = 1.0
            acc_mask[15] = 1.0

        is_log_step = (ep + 1) % 1000 == 0  # 로그 주기 조정

        try:
            res_base = sysid.run(true_data, raw_meas, temps, acc_mask=baseline_mask)
            mse_base = calculate_mse(res_base, true_data, raw_meas, temps)

            res_action = sysid.run(true_data, raw_meas, temps, acc_mask=acc_mask)
            mse_action = calculate_mse(res_action, true_data, raw_meas, temps)

            abs_improvement = mse_base - mse_action
            improvement_ratio = abs_improvement / (mse_base + 1e-9)

            fitness = 0.0
            if improvement_ratio < -0.01:
                fitness = -5.0
            elif abs_improvement < sensor_variance:
                fitness = 0.0
            else:
                fitness = improvement_ratio * 10.0

            # Cost: Bias=0.2, Temp=2.0
            cost = (0.2 * selected[0]) + (2.0 * selected[1])
            reward = fitness - cost

            if is_log_step:
                print(f"\n[Debug Ep {ep + 1}] Mode: {mode}")
                print(f"  > Action: {selected.astype(int)}")
                print(f"  > Ratio:  {improvement_ratio:.4f}")
                print(f"  > Rew: {reward:.2f}")

        except:
            reward = -10.0

        agent.update(log_prob, reward)
        reward_history.append(reward)

    print("\n[Test Results (Deterministic)]")
    for m in ["Stable", "Rising"]:
        _, _, ts = generate_temp_scenario(dt, duration, m)
        s = get_normalized_state(ts)

        a, _, raw_mu = agent.get_action(s, deterministic=True)
        decision = a > 0.0

        print(f"Scenario: {m} (Std: {np.std(ts):.2f})")
        print(f"  > Learn Bias?     : {'YES' if decision[0] else 'NO'}")
        print(f"  > Learn TempCoef? : {'YES' if decision[1] else 'NO'}")
        print(f"  > Raw Outputs (mu): {raw_mu}")  # 디버깅용 출력

        if m == "Stable" and decision[0] and not decision[1]:
            print("  => CORRECT Decision (Frugal)")
        elif m == "Rising" and decision[0] and decision[1]:
            print("  => CORRECT Decision (Effective)")
        else:
            print("  => INCORRECT Decision")
        print("-" * 30)


if __name__ == "__main__":
    main()
