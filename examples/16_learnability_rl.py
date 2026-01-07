import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import gtsam
import torch

from src.dynamics.ground import GroundVehicle
from src.sensors.imu import ImuSensor
from src.calibration.sysid_corrector import SysIdCalibrator
from src.calibration.rl_agent import RLAgent


# --- Helper: 랜덤 시나리오 생성 (확장됨) ---
def generate_random_maneuver(dt, duration):
    """
    랜덤한 주행 기동 생성
    [확장] Turn_Accel (가속 선회), Stop_and_Go (가감속) 추가
    """
    maneuver_list = ["Stop", "ConstVel", "Accel", "Turn", "Turn_Accel", "Stop_and_Go"]
    maneuver_type = np.random.choice(maneuver_list)

    vehicle = GroundVehicle()
    steps = int(duration / dt)

    # Kinematics buffers
    true_kinematics = []  # (specific_force_body, omega_body)

    vel_x = 0.0
    if maneuver_type in ["ConstVel", "Turn", "Turn_Accel"]:
        vel_x = 10.0

    for i in range(steps):
        t = i * dt
        acc_cmd = 0.0
        yaw_rate = 0.0

        # 시나리오별 명령 생성
        if maneuver_type == "Stop":
            vel_x = 0.0
            acc_cmd = 0.0
            yaw_rate = 0.0

        elif maneuver_type == "ConstVel":
            acc_cmd = 0.0
            yaw_rate = 0.0

        elif maneuver_type == "Accel":
            acc_cmd = 2.0 * np.sin(t)  # 가변 가속도
            vel_x += acc_cmd * dt

        elif maneuver_type == "Turn":
            yaw_rate = 0.5  # 정속 선회

        elif maneuver_type == "Turn_Accel":
            # 가속하면서 선회 (Misalignment 관측에 유리)
            acc_cmd = 1.5 * np.cos(t)
            vel_x += acc_cmd * dt
            yaw_rate = 0.5

        elif maneuver_type == "Stop_and_Go":
            # 급가속/급감속 반복
            acc_cmd = 3.0 * np.sign(np.sin(t))
            vel_x += acc_cmd * dt
            if vel_x < 0:
                vel_x = 0  # 후진 방지

        vehicle.update(dt, vel_x, yaw_rate)

        # True Data Generation (Specific Force)
        # SF = a_kinematic - g_body
        true_acc_kinematic = np.array([acc_cmd, yaw_rate * vel_x, 0.0])
        true_omega = np.array([0.0, 0.0, yaw_rate])

        rot_wb = vehicle.current_pose.rotation()
        g_body = rot_wb.unrotate(gtsam.Point3(0, 0, -9.81))

        # [수정] gtsam 버전에 따라 Point3 객체 또는 numpy array가 반환될 수 있음
        if isinstance(g_body, np.ndarray):
            g_body_np = g_body
        else:
            g_body_np = np.array([g_body.x(), g_body.y(), g_body.z()])

        sf_body = true_acc_kinematic - g_body_np
        true_kinematics.append((sf_body, true_omega))

    return maneuver_type, vehicle, true_kinematics


def main():
    print("=== RL-based Learnability Judgment (Full Parameter / Multi-Scenario) ===")

    dt = 0.1
    duration = 2.0  # 판단 윈도우
    episodes = 3000

    # 1. 고정된 True Error 설정 (Full Matrix)
    true_acc_bias = np.array([0.2, -0.1, 0.05])
    # Accel: Scale(대각) + Misalign(비대각)
    true_T_acc = np.array(
        [
            [1.05, 0.02, 0.00],  # X축 입력이 X, Y 출력에 영향
            [0.02, 1.03, 0.01],
            [0.00, 0.01, 0.98],
        ]
    )

    true_gyr_bias = np.array([0.01, 0.01, -0.01])
    # Gyro: Scale(대각) + Misalign(비대각)
    true_T_gyr = np.array(
        [
            [1.00, 0.00, 0.00],
            [0.00, 1.02, 0.02],  # Y, Z 간섭
            [0.00, 0.02, 1.01],
        ]
    )

    imu = ImuSensor(
        accel_noise=0.01,
        gyro_noise=0.001,
        accel_bias=true_acc_bias,
        gyro_bias=true_gyr_bias,
        accel_error_matrix=true_T_acc,
        gyro_error_matrix=true_T_gyr,
    )

    # 2. RL Agent (Selector)
    # Input: IMU Mean(6) + Std(6) = 12
    # Output: Accel Mask(12) + Gyro Mask(12) = 24 dimensions
    # Mask 구조: [T_row0(3), T_row1(3), T_row2(3), Bias(3)]
    agent = RLAgent(input_dim=12, action_dim=24, lr=0.001)

    sysid = SysIdCalibrator()
    reward_history = []

    print("Training Selector Agent...")

    for ep in range(episodes):
        # A. 랜덤 기동 생성
        m_type, vehicle, true_meas = generate_random_maneuver(dt, duration)

        # B. 센서 측정
        raw_meas = []
        for sf, om in true_meas:
            meas = imu.measure(gtsam.Pose3(), sf, om)
            raw_meas.append((meas[0], meas[1]))

        # C. State 추출
        raw_acc = np.array([m[0] for m in raw_meas])
        raw_gyr = np.array([m[1] for m in raw_meas])

        state = np.concatenate(
            [
                np.mean(raw_acc, axis=0),
                np.std(raw_acc, axis=0),
                np.mean(raw_gyr, axis=0),
                np.std(raw_gyr, axis=0),
            ]
        )

        # D. Action (24 dim) -> Masks
        action_raw, log_prob = agent.get_action(state)

        # Action Interpretation (Positive=Active)
        mask_threshold = 0.0
        selected = (action_raw > mask_threshold).astype(float)

        acc_mask = selected[0:12]  # Accel: Matrix(9) + Bias(3)
        gyr_mask = selected[12:24]  # Gyro: Matrix(9) + Bias(3)

        # E. Try Optimization
        try:
            res = sysid.run(true_meas, raw_meas, acc_mask=acc_mask, gyr_mask=gyr_mask)

            # F. Reward Calculation (Full Parameter)

            # 1) Accel Error Reduction
            acc_b_err_init = np.linalg.norm(true_acc_bias)
            acc_b_err_post = np.linalg.norm(true_acc_bias - res["acc_b"])

            acc_T_err_init = np.linalg.norm(true_T_acc - np.eye(3))
            # T_res = True * Est_Inv (Identity에 가까워야 함)
            T_res_acc = true_T_acc @ res["acc_T_inv"]
            acc_T_err_post = np.linalg.norm(T_res_acc - np.eye(3))

            gain_acc = (acc_b_err_init - acc_b_err_post) + (acc_T_err_init - acc_T_err_post)

            # 2) Gyro Error Reduction
            gyr_b_err_init = np.linalg.norm(true_gyr_bias)
            gyr_b_err_post = np.linalg.norm(true_gyr_bias - res["gyr_b"])

            gyr_T_err_init = np.linalg.norm(true_T_gyr - np.eye(3))
            T_res_gyr = true_T_gyr @ res["gyr_T_inv"]
            gyr_T_err_post = np.linalg.norm(T_res_gyr - np.eye(3))

            gain_gyr = (gyr_b_err_init - gyr_b_err_post) + (gyr_T_err_init - gyr_T_err_post)

            # 3) Sparsity Penalty
            num_active = np.sum(selected)
            penalty = 0.005 * num_active  # 페널티는 작게

            reward = (gain_acc * 10.0) + (gain_gyr * 10.0) - penalty

        except:
            reward = -2.0

        agent.update(log_prob, reward)
        reward_history.append(reward)

        if (ep + 1) % 500 == 0:
            print(f"Ep {ep + 1}: Avg Reward {np.mean(reward_history[-100:]):.4f}")

    # --- Test & Analyze Decisions ---
    print("\n=== Test Results: Scenario-based Parameter Selection ===")
    test_scenarios = ["Stop", "ConstVel", "Accel", "Turn", "Turn_Accel"]

    # 파라미터 이름 매핑 (인덱스 -> 의미)
    # Accel Matrix(0~8), Bias(9~11)
    # 0:xx, 1:xy, 2:xz, 3:yx, 4:yy ...
    param_names = [
        "Acc_Scale_X",
        "Acc_Mis_XY",
        "Acc_Mis_XZ",
        "Acc_Mis_YX",
        "Acc_Scale_Y",
        "Acc_Mis_YZ",
        "Acc_Mis_ZX",
        "Acc_Mis_ZY",
        "Acc_Scale_Z",
        "Acc_Bias_X",
        "Acc_Bias_Y",
        "Acc_Bias_Z",
    ]
    # Gyro는 12~23 인덱스 (위와 동일 구조)

    for sc in test_scenarios:
        _, _, t_meas = generate_random_maneuver(dt, duration)

        # Meas
        r_meas = []
        for sf, om in t_meas:
            m = imu.measure(gtsam.Pose3(), sf, om)
            r_meas.append((m[0], m[1]))

        r_acc = np.array([m[0] for m in r_meas])
        r_gyr = np.array([m[1] for m in r_meas])
        state = np.concatenate(
            [
                np.mean(r_acc, axis=0),
                np.std(r_acc, axis=0),
                np.mean(r_gyr, axis=0),
                np.std(r_gyr, axis=0),
            ]
        )

        action, _ = agent.get_action(state)
        judgment = action > 0.0

        print(f"\n[Scenario: {sc}]")

        # 주요 파라미터 몇 가지만 출력 확인
        print("  [Accel] Selected Params:")
        active_params = []
        for i in range(12):
            if judgment[i]:
                active_params.append(param_names[i])
        if not active_params:
            print("    (None)")
        else:
            print(f"    {', '.join(active_params)}")

        print("  [Gyro ] Selected Params:")
        # Gyro Bias Z (Idx 23), Scale Z (Idx 20) 확인
        if judgment[23]:
            print("    Gyr_Bias_Z: YES", end=" ")
        if judgment[20]:
            print("    Gyr_Scale_Z: YES", end=" ")
        print("")


if __name__ == "__main__":
    main()
