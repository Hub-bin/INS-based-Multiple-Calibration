import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import gtsam

from src.dynamics.ground import GroundVehicle
from src.sensors.imu import ImuSensor
from src.calibration.sysid_corrector import SysIdCalibrator
from src.calibration.rl_agent import RLAgent


def generate_hysteresis_scenario(dt, duration, mode="Cruising"):
    steps = int(duration / dt)
    vehicle = GroundVehicle()
    true_data = []
    temps = []  # 온도는 일정하다고 가정 (20도)

    vel_x = 0.0
    if mode == "Cruising":
        vel_x = 10.0

    for i in range(steps):
        t = i * dt
        acc_cmd = 0.0

        if mode == "Cruising":
            # 약간의 진동만 있음
            acc_cmd = 0.0 + np.random.normal(0, 0.05)
        elif mode == "Stop_and_Go":
            # 가속과 감속 반복 (+2.0 ~ -2.0)
            acc_cmd = 2.0 * np.sin(2.0 * t)

        vel_x += acc_cmd * dt
        if vel_x < 0:
            vel_x = 0

        vehicle.update(dt, vel_x, 0.0)
        temps.append(20.0)

        rot = vehicle.current_pose.rotation()
        g_body_np = rot.unrotate(gtsam.Point3(0, 0, -9.81))
        if not isinstance(g_body_np, np.ndarray):
            g_body_np = np.array([g_body_np.x(), g_body_np.y(), g_body_np.z()])

        true_acc_kin = np.array([acc_cmd, 0.0, 0.0])
        sf_body = true_acc_kin - g_body_np
        true_data.append((sf_body, np.zeros(3)))

    return vehicle, true_data, temps


def get_state(raw_acc_data):
    # State: [가속도 표준편차, 가속도 변화량의 평균 절대값(Jerk)]
    # 가속도 변화가 클수록 히스테리시스 관측 가능
    arr = np.array([x[0] for x in raw_acc_data])  # X축만

    std_val = np.std(arr)
    diff_val = np.mean(np.abs(np.diff(arr)))

    # Normalize
    return np.array([std_val / 2.0, diff_val / 0.5])


def calculate_mse(sysid_result, true_data, raw_data):
    mse_sum = 0.0
    count = len(raw_data)
    est_b = sysid_result["acc_b"]
    est_h = sysid_result["acc_h"]

    # Raw diff for hysteresis direction
    raw_acc_arr = np.array([r[0] for r in raw_data])
    raw_diff = np.diff(raw_acc_arr, axis=0, prepend=raw_acc_arr[0:1])
    hyst_sign = np.sign(raw_diff)

    for i in range(count):
        raw_acc = raw_data[i][0]
        true_acc = true_data[i][0]

        # Correction: Meas = True + Bias + H * sign
        # -> True = Meas - Bias - H * sign
        correction = est_b + (est_h * hyst_sign[i])
        est_true_acc = raw_acc - correction

        mse_sum += (true_acc[0] - est_true_acc[0]) ** 2

    return mse_sum / count


def main():
    print("=== RL: Hysteresis Learnability (Direction Dependent Bias) ===")

    dt = 0.1
    duration = 5.0
    episodes = 2500

    accel_noise = 0.002
    sensor_variance = accel_noise**2

    # 히스테리시스 오차 주입 (0.1 m/s^2)
    # Bias도 섞어줌
    imu = ImuSensor(
        accel_bias=[0.2, 0.0, 0.0], accel_hysteresis=0.1, accel_noise=accel_noise, gyro_noise=0.0001
    )

    # Action Dim 2: [Bias, Hysteresis]
    agent = RLAgent(input_dim=2, action_dim=2, lr=0.005)
    sysid = SysIdCalibrator()
    reward_history = []

    print("Training RL Agent...")

    # Baseline: Bias Only (Index 9~11)
    baseline_mask = np.zeros(21)
    baseline_mask[9:12] = 1.0

    for ep in range(episodes):
        mode = np.random.choice(["Cruising", "Stop_and_Go"])
        _, true_data, temps = generate_hysteresis_scenario(dt, duration, mode)

        raw_meas = []
        for i, (sf, om) in enumerate(true_data):
            m = imu.measure(gtsam.Pose3(), sf, om, temperature=temps[i])
            raw_meas.append((m[0], m[1]))

        # 가속도 데이터 기반 State
        raw_acc_only = [m[0] for m in raw_meas]
        state = get_state(raw_acc_only)

        # [수정 1] 반환값 2개로 수정 (action, log_prob)
        action, log_prob = agent.get_action(state, deterministic=False)
        selected = (action > 0.0).astype(float)

        acc_mask = np.zeros(21)
        if selected[0]:
            acc_mask[9] = 1.0  # Bias
        if selected[1]:
            acc_mask[18] = 1.0  # Hysteresis (Idx 18~20)

        is_log_step = (ep + 1) % 500 == 0

        try:
            # 1. Baseline Run
            res_base = sysid.run(true_data, raw_meas, temps, acc_mask=baseline_mask)
            mse_base = calculate_mse(res_base, true_data, raw_meas)

            # 2. Action Run
            res_action = sysid.run(true_data, raw_meas, temps, acc_mask=acc_mask)
            mse_action = calculate_mse(res_action, true_data, raw_meas)

            abs_improvement = mse_base - mse_action
            improvement_ratio = abs_improvement / (mse_base + 1e-9)

            fitness = 0.0

            if improvement_ratio < -0.01:
                fitness = -5.0
            elif abs_improvement < sensor_variance:
                fitness = 0.0
            else:
                fitness = improvement_ratio * 10.0

            # Cost Tuning
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
    for m in ["Cruising", "Stop_and_Go"]:
        # [수정 2] 올바른 변수 언패킹
        vehicle_test, true_data_test, temps_test = generate_hysteresis_scenario(dt, duration, m)

        # Test Measurement Generation
        rm = []
        for sf, om in true_data_test:
            meas = imu.measure(gtsam.Pose3(), sf, om, temperature=20.0)
            rm.append((meas[0], meas[1]))

        raw_acc_only = [x[0] for x in rm]
        s = get_state(raw_acc_only)

        # [수정 3] 반환값 2개로 수정 (action, _)
        a, _ = agent.get_action(s, deterministic=True)
        decision = a > 0.0

        print(f"Scenario: {m}")
        print(f"  > Learn Bias? : {'YES' if decision[0] else 'NO'}")
        print(f"  > Learn Hyst? : {'YES' if decision[1] else 'NO'}")

        if m == "Cruising" and not decision[1]:
            print("  => CORRECT (Frugal)")
        elif m == "Stop_and_Go" and decision[1]:
            print("  => CORRECT (Effective)")
        else:
            print("  => INCORRECT")
        print("-" * 30)


if __name__ == "__main__":
    main()
