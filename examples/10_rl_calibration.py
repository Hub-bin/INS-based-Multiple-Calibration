import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gtsam
import matplotlib.pyplot as plt

from src.dynamics.ground import GroundVehicle
from src.sensors.imu import ImuSensor
from src.calibration.rl_agent import RLAgent
from src.utils.evaluation import calculate_rmse


# 환경(Environment) 역할을 하는 함수
def run_environment_episode(vehicle, imu, duration, dt, action_params):
    """
    1. 시뮬레이션 데이터 생성 (매 에피소드마다 노이즈 랜덤)
    2. RL Action(파라미터) 적용하여 데이터 보정
    3. 항법(Dead Reckoning) 수행
    4. Reward 계산
    """
    # Action 분해
    pred_acc_bias, pred_gyr_bias, pred_T_acc = action_params

    # 데이터 생성 (매번 랜덤한 노이즈 상황)
    steps = int(duration / dt)
    velocity_x = 10.0
    measurements = []

    # 차량 초기화 (위치 0,0,0)
    vehicle.current_pose = gtsam.Pose3()
    vehicle.poses = [vehicle.current_pose]

    # IMU 센서 모델은 고정된 True Error를 가짐 (외부 환경)
    # 하지만 측정값에는 랜덤 노이즈가 매번 다르게 섞임

    for i in range(steps):
        yaw_rate = 0.5 * np.sin(i * dt * 0.5)  # S자 주행

        # Ground Truth Kinematics
        true_accel = np.array([0.0, yaw_rate * velocity_x, 0.0])
        true_omega = np.array([0.0, 0.0, yaw_rate])

        vehicle.update(dt, velocity_x, yaw_rate)

        # Raw Measurement 생성 (오차 포함)
        meas = imu.measure(vehicle.current_pose, true_accel, true_omega)

        # --- [RL Agent의 개입] ---
        # Agent가 추정한 파라미터로 "역보정" 수행
        # Corrected = T_pred_inv * (Raw - Bias_pred)
        # 여기서는 간단히 행렬 곱과 뺄셈으로 구현

        raw_acc, raw_gyr = meas

        # 1. Bias 제거
        corr_acc = raw_acc - pred_acc_bias
        corr_gyr = raw_gyr - pred_gyr_bias

        # 2. Scale/Misalign 제거 (Inverse Matrix 적용)
        # Agent가 출력한게 T_acc의 "역행렬"이라고 가정하고 학습시킴 (그게 더 쉬움)
        corr_acc = pred_T_acc @ corr_acc

        measurements.append((corr_acc, corr_gyr))

    # 항법 수행 (GTSAM Preintegration or Dead Reckoning)
    # RL이 잘했으면 Bias/Scale이 사라졌을 것이므로 Identity Bias로 적분
    pim_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
    zero_bias = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
    pim = gtsam.PreintegratedImuMeasurements(pim_params, zero_bias)

    est_poses = [vehicle.poses[0]]
    curr_pose = vehicle.poses[0]
    curr_vel = gtsam.Point3(velocity_x, 0, 0)

    for acc, gyr in measurements:
        pim.integrateMeasurement(acc, gyr, dt)
        nav_state = gtsam.NavState(curr_pose, curr_vel)
        next_state = pim.predict(nav_state, zero_bias)
        curr_pose = next_state.pose()
        curr_vel = next_state.velocity()
        est_poses.append(curr_pose)
        pim.resetIntegration()

    # RMSE 계산
    rmse_pos, _ = calculate_rmse(vehicle.poses, est_poses)

    # Reward 설계
    # RMSE가 작을수록 보상이 커야 함. (예: e^-error 또는 1/error)
    # 학습 안정성을 위해 음의 RMSE 사용하되, 너무 크면 클리핑
    reward = -rmse_pos

    return reward, rmse_pos


def main():
    print("=== Reinforcement Learning Calibration (Policy Gradient) ===")

    dt = 0.1
    duration = 20.0
    episodes = 2000  # 학습 횟수

    # 1. 환경 설정 (고정된 True Error, Agent는 이걸 모름)
    true_acc_bias = np.array([0.2, -0.1, 0.05])
    true_gyr_bias = np.array([0.01, 0.02, -0.01])
    # Scale: 1.05, Misalign: 0.02
    true_T_acc = np.array([[1.05, 0.02, 0.0], [0.02, 1.05, 0.0], [0.0, 0.0, 1.0]])

    vehicle = GroundVehicle()
    imu = ImuSensor(
        accel_noise=0.02,
        gyro_noise=0.005,
        accel_bias=true_acc_bias,
        gyro_bias=true_gyr_bias,
        accel_error_matrix=true_T_acc,
    )

    # 2. RL Agent 생성
    # State: [Avg_Acc_X, Avg_Acc_Y, Avg_Acc_Z, Std_Acc_X... Avg_Gyr... Std_Gyr...] -> 12 dim
    # Action: [Acc_Bias(3), Gyr_Bias(3), T_Matrix(9)] -> 15 dim
    agent = RLAgent(input_dim=12, action_dim=15, lr=0.001)

    reward_history = []
    rmse_history = []

    print(f"Starting Training for {episodes} episodes...")

    for ep in range(episodes):
        # --- State 관측 (Observation) ---
        # 학습을 위해 짧게 데이터를 수집하여 통계적 특징 추출
        # 실제로는 "현재 윈도우의 센서 데이터 통계"가 State가 됨
        obs_duration = 2.0
        obs_steps = int(obs_duration / dt)
        raw_obs = []
        for _ in range(obs_steps):
            # 정지 상태 혹은 단순 주행 상태라고 가정 (State 추출용)
            meas = imu.measure(gtsam.Pose3(), np.zeros(3), np.zeros(3))
            raw_obs.append(meas)

        acc_data = np.array([m[0] for m in raw_obs])
        gyr_data = np.array([m[1] for m in raw_obs])

        # Feature Extraction (Mean, Std)
        state = np.concatenate(
            [
                np.mean(acc_data, axis=0),
                np.std(acc_data, axis=0),
                np.mean(gyr_data, axis=0),
                np.std(gyr_data, axis=0),
            ]
        )

        # --- Action (Exploration) ---
        action_vector, log_prob = agent.get_action(state)
        action_params = agent.decode_action(action_vector)

        # --- Environment Step ---
        reward, rmse = run_environment_episode(vehicle, imu, duration, dt, action_params)

        # --- Update (Learn) ---
        # Baseline을 빼주는 것이 좋지만 여기선 단순 REINFORCE
        loss = agent.update(log_prob, reward)

        reward_history.append(reward)
        rmse_history.append(rmse)

        if (ep + 1) % 100 == 0:
            avg_rmse = np.mean(rmse_history[-100:])
            print(f"Episode {ep + 1}: Avg RMSE = {avg_rmse:.4f} m, Last Reward = {reward:.2f}")

    # 3. 결과 시각화
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rmse_history)
    plt.title("Training Progress (RMSE)")
    plt.xlabel("Episode")
    plt.ylabel("Position Error (m)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    # 마지막 에피소드의 파라미터로 검증
    final_state = state  # 마지막 state 사용
    final_action, _ = agent.get_action(final_state)
    final_params = agent.decode_action(final_action)

    print("\nTraining Finished.")
    print("Final Learned Params:")
    print("  Pred Acc Bias:", final_params[0])
    print("  Pred Gyr Bias:", final_params[1])
    # T_matrix는 역행렬을 학습했으므로 원래 T와 곱하면 I가 나와야 함

    plt.show()


if __name__ == "__main__":
    main()
