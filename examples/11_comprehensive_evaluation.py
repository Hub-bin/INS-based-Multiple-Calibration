import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gtsam

from src.dynamics.ground import GroundVehicle
from src.sensors.imu import ImuSensor
from src.calibration.offline import OfflineCalibrator
from src.calibration.online import OnlineCalibrator  # [New] Online 모듈 추가 필요
from src.calibration.rl_agent import RLAgent
from src.utils.advanced_viz import plot_comprehensive_dashboard


# --- Helper Functions ---
def generate_simulation_data(dt, duration, vehicle, imu):
    """시뮬레이션 데이터 생성 (GT, Raw Meas) - 가감속 추가 버전"""
    steps = int(duration / dt)

    # [수정] 초기 속도 및 위치
    current_velocity_x = 10.0  # 초기 속도

    raw_measurements = []

    # Reset vehicle
    vehicle.current_pose = gtsam.Pose3()
    vehicle.poses = [vehicle.current_pose]

    for i in range(steps):
        t = i * dt

        # 1. 조향 (S-Curve)
        yaw_rate = 0.5 * np.sin(t * 0.5)

        # 2. [핵심 수정] 가감속 추가 (Longitudinal Acceleration)
        # 2초 주기로 가속/감속 반복 (최대 가속도 2m/s^2)
        # 이 가속도 성분(ax)이 있어야 Accel Bias와 Scale을 구분 가능함
        accel_x_cmd = 2.0 * np.cos(t * 0.5)

        # 속도 업데이트 (v = v0 + a*dt)
        current_velocity_x += accel_x_cmd * dt

        # 3. True Kinematics (Body Frame)
        # 가속도계가 느끼는 진짜 가속도 = 선가속도(ax) + 원심력(ay)
        # ax = accel_x_cmd
        # ay = yaw_rate * current_velocity_x
        true_acc = np.array([accel_x_cmd, yaw_rate * current_velocity_x, 0.0])
        true_gyr = np.array([0.0, 0.0, yaw_rate])

        # 4. Vehicle Pose 업데이트
        # GroundVehicle update 함수는 '속도' 입력을 받으므로 현재 속도 전달
        vehicle.update(dt, current_velocity_x, yaw_rate)

        # Measure
        meas = imu.measure(vehicle.current_pose, true_acc, true_gyr)
        raw_measurements.append(meas)

    return raw_measurements


def run_navigation_loop(start_pose, measurements, bias, dt, correction_matrix=None, pred_bias=None):
    """항법(Dead Reckoning) 수행"""
    pim_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
    pim = gtsam.PreintegratedImuMeasurements(pim_params, bias)  # GTSAM Bias

    poses = [start_pose]
    curr_pose = start_pose
    curr_vel = gtsam.Point3(10.0, 0, 0)

    for raw_acc, raw_gyr in measurements:
        # RL Correction 적용 (있을 경우)
        if correction_matrix is not None and pred_bias is not None:
            # 1. Bias removal
            acc = raw_acc - pred_bias[0]
            gyr = raw_gyr - pred_bias[1]
            # 2. Scale/Misalign removal
            acc = correction_matrix @ acc
        else:
            acc, gyr = raw_acc, raw_gyr

        pim.integrateMeasurement(acc, gyr, dt)

        nav_state = gtsam.NavState(curr_pose, curr_vel)
        next_state = pim.predict(nav_state, bias)
        curr_pose = next_state.pose()
        curr_vel = next_state.velocity()
        poses.append(curr_pose)
        pim.resetIntegration()

    return poses


def calculate_errors(gt_poses, est_poses):
    """시간별 위치 오차 계산"""
    n = min(len(gt_poses), len(est_poses))
    errors = []
    for i in range(n):
        diff = gt_poses[i].translation() - est_poses[i].translation()
        errors.append(np.linalg.norm(diff))
    return errors


# --- Main ---
def main():
    print("=== Comprehensive Calibration Evaluation Dashboard ===")

    dt = 0.1
    duration = 30.0

    # [수정 1] 학습 횟수 대폭 증가 (500 -> 3000)
    # 강화학습은 수렴에 시간이 걸리므로 충분한 에피소드가 필수적입니다.
    train_episodes = 10000

    # 1. 환경 설정 (복합 오차)
    true_acc_bias = np.array([0.2, -0.1, 0.05])
    true_gyr_bias = np.array([0.01, 0.02, -0.01])

    # Scale Factor (대각선) & Misalignment (비대각선)
    true_T_acc = np.array([[1.05, 0.02, 0.01], [0.02, 1.03, 0.01], [0.00, 0.01, 0.98]])

    vehicle = GroundVehicle()
    imu = ImuSensor(
        accel_noise=0.02,
        gyro_noise=0.005,
        accel_bias=true_acc_bias,
        gyro_bias=true_gyr_bias,
        accel_error_matrix=true_T_acc,
    )

    # 2. RL Agent 학습
    print(f"1. Training RL Agent ({train_episodes} episodes)...")
    # Learning Rate를 약간 낮춰서 안정적 수렴 유도
    agent = RLAgent(input_dim=12, action_dim=15, lr=0.001)

    for ep in range(train_episodes):
        # 짧은 데이터로 학습
        _ = generate_simulation_data(dt, 5.0, vehicle, imu)

        # State Extraction
        meas_sample = imu.measure(gtsam.Pose3(), np.zeros(3), np.zeros(3))
        state = np.concatenate(
            [meas_sample[0], meas_sample[0] * 0.1, meas_sample[1], meas_sample[1] * 0.1]
        )

        action_vec, log_prob = agent.get_action(state)
        acc_b, gyr_b, T_inv = agent.decode_action(action_vec)

        # Reward: 파라미터 오차 최소화 (Supervised guided RL for Demo)
        err_b = np.linalg.norm(acc_b - true_acc_bias)
        # RL이 추정한 T_inv와 실제 T를 곱했을 때 Identity가 되어야 함
        res = T_inv @ true_T_acc
        err_identity = np.linalg.norm(res - np.eye(3))

        # Scale/Misalign 오차에 가중치를 더 둠 (GTSAM이 못 잡는 부분이므로)
        reward = -(err_b + err_identity * 5.0)
        agent.update(log_prob, reward)

        if (ep + 1) % 500 == 0:
            print(f"  Ep {ep + 1}: Reward {reward:.2f} (Matrix Err: {err_identity:.4f})")

    print("Training Done.")

    # 3. 평가 데이터 생성
    print("2. Generating Evaluation Data...")
    raw_measurements = generate_simulation_data(dt, duration, vehicle, imu)
    gt_poses = vehicle.poses

    # 4. Conventional Method (GTSAM Only) - Online으로 변경하여 비교
    print("3. Running Conventional Method (Online)...")

    # Scale 오차를 모르는 상태에서 Bias만 Online으로 추정 시도
    online_calib_conv = OnlineCalibrator()
    online_calib_conv.initialize(gt_poses[0], gtsam.Point3(10, 0, 0))  # 초기화

    conv_poses = [gt_poses[0]]
    conv_bias_history = []

    for i, (acc, gyr) in enumerate(raw_measurements):
        # 현재 Pose(GPS)가 들어온다고 가정하고 Update
        # (실제론 GPS가 가끔 들어오지만 여기선 매 스텝 보정 효과 확인)
        est_bias = online_calib_conv.update(gt_poses[i], acc, gyr, dt)
        conv_bias_history.append(est_bias)

        # 항법은 편의상 Offline 로직 재사용하거나 별도 루프 필요하나,
        # 여기선 OnlineCalibrator 내부 상태가 추정된 Bias임.
        # 시각화를 위해 마지막에 한 번에 다시 적분하거나, 위 update 루프 내에서 적분해야 함.
        pass

    # 공정한 비교를 위해 추정된 '평균 Bias'로 전체 궤적 재생성 (또는 실시간 궤적 저장)
    # 여기서는 '최종 수렴된 Bias'를 사용하여 전체 궤적을 다시 그립니다.
    final_bias_conv = conv_bias_history[-1]
    conv_poses = run_navigation_loop(gt_poses[0], raw_measurements, final_bias_conv, dt)
    conv_errors = calculate_errors(gt_poses, conv_poses)

    # 5. Proposed Method (RL + Online GTSAM)
    print("4. Running Proposed Method (RL + Online)...")

    # A. RL Inference (Pre-correction Parameter 추론)
    state_sample = np.concatenate(
        [
            raw_measurements[0][0],
            raw_measurements[0][0] * 0.1,
            raw_measurements[0][1],
            raw_measurements[0][1] * 0.1,
        ]
    )
    action_vec, _ = agent.get_action(state_sample)
    pred_acc_b, pred_gyr_b, pred_T_inv = agent.decode_action(action_vec)

    # B. Online Calibration Loop
    # RL이 Scale/Misalign을 잡았으므로, GTSAM은 잔여 Bias만 잡으면 됨
    online_calib_prop = OnlineCalibrator()
    online_calib_prop.initialize(gt_poses[0], gtsam.Point3(10, 0, 0))

    prop_bias_history = []

    for i, (ra, rg) in enumerate(raw_measurements):
        # 1. RL Correction
        ca = pred_T_inv @ (ra - pred_acc_b)
        cg = rg - pred_gyr_b

        # 2. GTSAM Update (Corrected Data 입력)
        est_bias_resid = online_calib_prop.update(gt_poses[i], ca, cg, dt)
        prop_bias_history.append(est_bias_resid)

    # C. 최종 궤적 생성
    final_bias_resid = prop_bias_history[-1]
    # RL 파라미터와 GTSAM 잔여 Bias를 모두 적용
    rl_poses = run_navigation_loop(
        gt_poses[0],
        raw_measurements,
        final_bias_resid,
        dt,
        correction_matrix=pred_T_inv,
        pred_bias=(pred_acc_b, pred_gyr_b),
    )
    rl_errors = calculate_errors(gt_poses, rl_poses)

    # 6. 결과 시각화
    print("5. Visualizing Results...")

    traj_dict = {"GT": gt_poses, "Conv": conv_poses, "RL": rl_poses}
    error_dict = {"Conv": conv_errors, "RL": rl_errors}

    bias_dict = {
        "True": (true_acc_bias, true_gyr_bias),
        "Conv": (final_bias_conv.accelerometer(), final_bias_conv.gyroscope()),
        "RL": (
            pred_acc_b + final_bias_resid.accelerometer(),
            pred_gyr_b + final_bias_resid.gyroscope(),
        ),
    }

    matrix_dict = {"True_T_acc": true_T_acc, "RL_T_acc_inv": pred_T_inv}

    plot_comprehensive_dashboard(traj_dict, error_dict, bias_dict, matrix_dict)


if __name__ == "__main__":
    main()
