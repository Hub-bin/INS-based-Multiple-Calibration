import sys
import os

# 부모 디렉토리를 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gtsam
import matplotlib.pyplot as plt

from src.dynamics.ground import GroundVehicle
from src.sensors.imu import ImuSensor
from src.calibration.offline import OfflineCalibrator
from src.calibration.ai_corrector import AiCalibrator
from src.utils.evaluation import calculate_rmse, plot_trajectory_comparison, plot_error_analysis
from src.utils.map_viz import RealMapVisualizer


def main():
    print("=== Advanced AI Calibration: Bias, Scale, Misalignment ===")

    dt = 0.1
    sim_duration = 30.0  # 충분한 데이터 확보

    # 1. 복합 오차 설정 (정답)
    # Bias
    true_acc_bias = np.array([0.2, -0.1, 0.05])
    true_gyr_bias = np.array([0.01, 0.02, -0.01])

    # Scale Factor & Misalignment Matrix (T)
    # 대각선 1.05 -> 5% 스케일 오차
    # 비대각선 0.02 -> 축 비정렬 오차
    true_acc_T = np.array([[1.05, 0.02, 0.00], [0.02, 1.03, 0.01], [0.00, 0.01, 0.98]])
    true_gyr_T = np.array([[0.99, 0.01, 0.00], [0.01, 1.01, 0.02], [0.00, 0.00, 1.00]])

    print("True Accel Error Matrix (T):\n", true_acc_T)

    # 센서 생성
    vehicle = GroundVehicle()
    imu = ImuSensor(
        accel_noise=0.02,
        gyro_noise=0.005,
        accel_bias=true_acc_bias,
        gyro_bias=true_gyr_bias,
        accel_error_matrix=true_acc_T,
        gyro_error_matrix=true_gyr_T,
    )

    print("\n1. Generating Complex Trajectory (S-Curve)...")
    steps = int(sim_duration / dt)
    velocity_x = 10.0

    raw_measurements = []
    gt_measurements = []  # AI 학습용 Ground Truth (Noise 없는 이상적 값)

    for i in range(steps):
        yaw_rate = 0.5 * np.sin(i * dt * 0.5)

        # Ground Truth 물리량
        rot_wb = vehicle.current_pose.rotation()
        gravity = np.array([0, 0, -9.81])
        # Unrotate gravity manually for GT gen (simplified)
        # Note: ImuSensor logic uses pose to remove gravity.
        # For training targets, we want: Accel_body(Kinematic) - Unrotated_Gravity

        # 간단히 ImuSensor의 내부 로직을 역이용하거나, measure 호출 시 True값을 저장
        true_accel_kinematic = np.array([0.0, yaw_rate * velocity_x, 0.0])
        true_omega = np.array([0.0, 0.0, yaw_rate])

        # 센서 측정 (Noise + Bias + Scale + Misalign)
        vehicle.update(dt, velocity_x, yaw_rate)
        meas = imu.measure(vehicle.current_pose, true_accel_kinematic, true_omega)
        raw_measurements.append(meas)

        # AI 학습용 정답 데이터 (Specific Force & Omega without Error)
        # measure 함수 내부 로직과 동일하게 중력 반영
        g_body = rot_wb.unrotate(gtsam.Point3(*gravity))
        if not isinstance(g_body, np.ndarray):
            g_body = np.array([g_body.x(), g_body.y(), g_body.z()])

        gt_sf = true_accel_kinematic - g_body
        gt_measurements.append((gt_sf, true_omega))

    # 2. AI 선행 교정 (System Identification)
    print("\n2. [AI Phase] Learning Calibration Parameters (Inverse Model)...")
    ai_calib = AiCalibrator()
    # 시뮬레이션 환경이므로 GT를 사용하여 센서 특성을 학습
    ai_calib.train_simulation(gt_measurements, raw_measurements, epochs=1500)

    # 학습된 파라미터 확인
    params = ai_calib.get_calibration_params()
    print("   -> Learned Accel Bias:", params["accel_bias"])
    print("   -> Learned Accel Inv Matrix:\n", params["accel_matrix_inv"])
    print("   -> True Inverse (Target):\n", np.linalg.inv(true_acc_T))

    # 3. 데이터 보정 (Correction)
    print("\n3. Correcting Raw Data using AI...")
    corrected_measurements = ai_calib.correct(raw_measurements)

    # 4. GTSAM Navigation (보정된 데이터 사용)
    # 데이터가 이미 보정되었으므로, GTSAM에는 Bias=0으로 시작하도록 설정하거나
    # 아주 작은 잔여 Bias만 추정하도록 함.
    print("\n4. [Navigation Phase] Running GTSAM with Corrected Data...")

    # AI가 Bias까지 제거했으므로 초기 Bias는 0으로 가정
    zero_bias = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))

    # Offline Calibrator는 Bias 추정기가 포함되어 있으므로, 잔여 오차를 잡도록 둠
    offline_calib = OfflineCalibrator(init_bias=zero_bias)

    # AI 보정 데이터를 넣음
    final_bias = offline_calib.run(vehicle.poses, corrected_measurements, dt)
    print("   -> GTSAM Residual Bias Estimate:", final_bias)

    # 5. 성능 평가
    print("\n5. Evaluating Performance...")

    # 보정된 데이터로 궤적 적분
    pim_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
    # AI가 Scale/Misalign을 잡았으므로 Identity Covariance 사용 가능

    pim = gtsam.PreintegratedImuMeasurements(pim_params, final_bias)

    est_poses = [vehicle.poses[0]]
    curr_pose = vehicle.poses[0]
    curr_vel = gtsam.Point3(velocity_x, 0, 0)

    for i in range(len(corrected_measurements)):
        acc, gyr = corrected_measurements[i]
        pim.integrateMeasurement(acc, gyr, dt)

        nav_state = gtsam.NavState(curr_pose, curr_vel)
        next_state = pim.predict(nav_state, final_bias)

        curr_pose = next_state.pose()
        curr_vel = next_state.velocity()
        est_poses.append(curr_pose)
        pim.resetIntegration()

    rmse_pos, rmse_rot = calculate_rmse(vehicle.poses, est_poses)
    print(f"   -> Position RMSE: {rmse_pos:.4f} m")

    plot_trajectory_comparison(
        vehicle.poses, est_poses, title="AI Corrected (Bias+Scale+Misalign) Trajectory"
    )


if __name__ == "__main__":
    main()
