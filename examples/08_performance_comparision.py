import sys
import os

# 부모 디렉토리 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import gtsam

from src.dynamics.ground import GroundVehicle
from src.sensors.imu import ImuSensor
from src.calibration.offline import OfflineCalibrator
from src.calibration.ai_corrector import AiCalibrator
from src.utils.evaluation import calculate_rmse


def run_navigation(vehicle, measurements, bias_estimate, dt):
    """주어진 측정값과 Bias 추정치로 궤적(Dead Reckoning) 계산"""
    pim_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
    pim = gtsam.PreintegratedImuMeasurements(pim_params, bias_estimate)

    estimated_poses = [vehicle.poses[0]]
    current_pose = vehicle.poses[0]
    # 초기 속도 (x방향 10m/s 가정)
    current_vel = gtsam.Point3(10.0, 0, 0)

    for i in range(len(measurements)):
        acc, gyr = measurements[i]
        pim.integrateMeasurement(acc, gyr, dt)

        nav_state = gtsam.NavState(current_pose, current_vel)
        next_state = pim.predict(nav_state, bias_estimate)

        current_pose = next_state.pose()
        current_vel = next_state.velocity()
        estimated_poses.append(current_pose)

        pim.resetIntegration()

    return estimated_poses


def main():
    print("=== Performance Comparison: Conventional(GTSAM only) vs Proposed(AI + GTSAM) ===")

    dt = 0.1
    sim_duration = 30.0

    # 1. 시뮬레이션 환경 설정 (심한 오차 주입)
    # Bias
    true_acc_bias = np.array([0.2, -0.1, 0.05])
    true_gyr_bias = np.array([0.01, 0.02, -0.01])

    # Scale Factor (5% Error) & Misalignment
    error_matrix = np.array([[1.05, 0.02, 0.00], [0.02, 1.05, 0.01], [0.00, 0.01, 0.95]])

    vehicle = GroundVehicle()
    imu = ImuSensor(
        accel_noise=0.02,
        gyro_noise=0.005,
        accel_bias=true_acc_bias,
        gyro_bias=true_gyr_bias,
        accel_error_matrix=error_matrix,
        gyro_error_matrix=error_matrix,
    )

    print(f"1. Generating Data (Duration: {sim_duration}s)...")
    steps = int(sim_duration / dt)
    velocity_x = 10.0

    raw_measurements = []
    gt_measurements = []

    for i in range(steps):
        # S자 주행
        yaw_rate = 0.5 * np.sin(i * dt * 0.5)

        # Ground Truth (for AI training target)
        rot_wb = vehicle.current_pose.rotation()
        gravity = np.array([0, 0, -9.81])
        true_accel_kinematic = np.array([0.0, yaw_rate * velocity_x, 0.0])
        true_omega = np.array([0.0, 0.0, yaw_rate])

        vehicle.update(dt, velocity_x, yaw_rate)

        # 측정 (Raw Data with Errors)
        meas = imu.measure(vehicle.current_pose, true_accel_kinematic, true_omega)
        raw_measurements.append(meas)

        # 정답 (GT Data for Training)
        g_body = rot_wb.unrotate(gtsam.Point3(*gravity))
        if not isinstance(g_body, np.ndarray):
            g_body = np.array([g_body.x(), g_body.y(), g_body.z()])
        gt_sf = true_accel_kinematic - g_body
        gt_measurements.append((gt_sf, true_omega))

    # ---------------------------------------------------------
    # Case 1: Without AI (Conventional)
    # ---------------------------------------------------------
    print("\n2. [Case 1] Running Conventional Calibration (GTSAM only)...")
    # 기존 방식은 Scale/Misalignment를 모르므로, 단순히 Bias만 추정하려고 시도함
    init_bias_zero = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
    offline_calib_conv = OfflineCalibrator(init_bias=init_bias_zero)

    # 오차가 포함된 Raw 데이터를 그대로 사용
    est_bias_conv = offline_calib_conv.run(vehicle.poses, raw_measurements, dt)
    print("   -> Estimated Bias (Conventional):", est_bias_conv)

    # 항법 수행
    poses_conv = run_navigation(vehicle, raw_measurements, est_bias_conv, dt)
    rmse_conv_pos, _ = calculate_rmse(vehicle.poses, poses_conv)
    print(f"   -> Position RMSE: {rmse_conv_pos:.4f} m")

    # ---------------------------------------------------------
    # Case 2: With AI (Proposed)
    # ---------------------------------------------------------
    print("\n3. [Case 2] Running Proposed Calibration (AI + GTSAM)...")
    ai_calib = AiCalibrator()
    # AI 학습 (Pre-calibration)
    ai_calib.train_simulation(gt_measurements, raw_measurements, epochs=500)

    # 데이터 보정
    corrected_measurements = ai_calib.correct(raw_measurements)

    # GTSAM 실행 (보정된 데이터 사용)
    offline_calib_prop = OfflineCalibrator(init_bias=init_bias_zero)
    est_bias_prop = offline_calib_prop.run(vehicle.poses, corrected_measurements, dt)
    print("   -> Estimated Residual Bias (Proposed):", est_bias_prop)

    # 항법 수행
    poses_prop = run_navigation(vehicle, corrected_measurements, est_bias_prop, dt)
    rmse_prop_pos, _ = calculate_rmse(vehicle.poses, poses_prop)
    print(f"   -> Position RMSE: {rmse_prop_pos:.4f} m")

    # ---------------------------------------------------------
    # 4. Visualization
    # ---------------------------------------------------------
    print("\n4. Visualizing Comparison...")

    # Trajectory Plot
    plt.figure(figsize=(12, 6))

    gt_x = [p.x() for p in vehicle.poses]
    gt_y = [p.y() for p in vehicle.poses]
    plt.plot(gt_x, gt_y, "k--", linewidth=2, label="Ground Truth")

    conv_x = [p.x() for p in poses_conv]
    conv_y = [p.y() for p in poses_conv]
    plt.plot(conv_x, conv_y, "r-", label=f"Without AI (RMSE: {rmse_conv_pos:.2f}m)")

    prop_x = [p.x() for p in poses_prop]
    prop_y = [p.y() for p in poses_prop]
    plt.plot(prop_x, prop_y, "b-", linewidth=2, label=f"With AI (RMSE: {rmse_prop_pos:.2f}m)")

    plt.title("Navigation Performance Comparison")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    # Error Bar Chart
    plt.figure(figsize=(6, 6))
    plt.bar(["Without AI", "With AI"], [rmse_conv_pos, rmse_prop_pos], color=["red", "blue"])
    plt.title("Position RMSE Comparison")
    plt.ylabel("RMSE (m)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.show()


if __name__ == "__main__":
    main()
