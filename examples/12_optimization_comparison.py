import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gtsam
import matplotlib.pyplot as plt

from src.dynamics.ground import GroundVehicle
from src.sensors.imu import ImuSensor
from src.calibration.offline import OfflineCalibrator
from src.calibration.sysid_corrector import SysIdCalibrator
from src.utils.evaluation import calculate_rmse, plot_trajectory_comparison
from src.utils.advanced_viz import plot_comprehensive_dashboard


def generate_stop_and_go_data(dt, duration, vehicle, imu):
    """
    Stop & Go 패턴 + S-Curve
    - 가속도계의 관측성(Observability)을 극대화하기 위해
      정지(0g, -9.8g)와 가속(+ax) 구간을 반복
    """
    steps = int(duration / dt)
    current_velocity_x = 0.0

    raw_measurements = []
    gt_measurements = []  # (True Accel, True Gyro)

    vehicle.current_pose = gtsam.Pose3()
    vehicle.poses = [vehicle.current_pose]

    for i in range(steps):
        t = i * dt

        # 시나리오: 2초 가속 -> 3초 정속 -> 2초 감속 -> 2초 정지 (반복)
        cycle_time = t % 9.0

        if cycle_time < 2.0:
            acc_cmd = 3.0  # 급가속
        elif cycle_time < 5.0:
            acc_cmd = 0.0  # 정속
        elif cycle_time < 7.0:
            acc_cmd = -3.0  # 급감속
        else:
            acc_cmd = 0.0  # 정지
            current_velocity_x = 0.0  # 강제 정지 처리

        # 속도 업데이트
        current_velocity_x += acc_cmd * dt
        if current_velocity_x < 0:
            current_velocity_x = 0

        # 조향 (가속할 때만 회전)
        if current_velocity_x > 0.1:
            yaw_rate = 0.5 * np.sin(t * 0.5)
        else:
            yaw_rate = 0.0

        # True Kinematics
        true_acc_kinematic = np.array([acc_cmd, yaw_rate * current_velocity_x, 0.0])
        true_omega = np.array([0.0, 0.0, yaw_rate])

        # Update Vehicle
        vehicle.update(dt, current_velocity_x, yaw_rate)

        # Measure (Raw)
        meas = imu.measure(vehicle.current_pose, true_acc_kinematic, true_omega)
        raw_measurements.append(meas)

        # Ground Truth for Optimization
        # [중요] 최적화 타겟은 "중력이 포함되지 않은 Kinematic Accel"이 아니라
        # IMU가 실제로 느껴야 하는 "중력이 포함된 Specific Force"여야 함.
        # ImuSensor는 내부적으로 gravity를 뺌 (Accel = a - g).
        # 따라서 최적화의 정답(Target)도 (a - g)여야 함.

        rot_wb = vehicle.current_pose.rotation()
        g_world = gtsam.Point3(0, 0, -9.81)
        g_body = rot_wb.unrotate(g_world)
        if not isinstance(g_body, np.ndarray):
            g_body = np.array([g_body.x(), g_body.y(), g_body.z()])

        # Target = a_body - g_body
        gt_sf = true_acc_kinematic - g_body
        gt_measurements.append((gt_sf, true_omega))

    return raw_measurements, gt_measurements


def run_navigation(start_pose, measurements, bias, dt, params=None):
    pim_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
    pim = gtsam.PreintegratedImuMeasurements(pim_params, bias)

    poses = [start_pose]
    curr_pose = start_pose
    curr_vel = gtsam.Point3(0, 0, 0)  # 정지 상태 출발

    for raw_acc, raw_gyr in measurements:
        if params is not None:
            # SysID Correction
            acc = params["acc_T_inv"] @ (raw_acc - params["acc_b"])
            gyr = params["gyr_T_inv"] @ (raw_gyr - params["gyr_b"])
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


def main():
    print("=== Optimization-based Calibration (System Identification) ===")

    dt = 0.05
    duration = 40.0  # 충분한 가감속 반복

    # 1. 환경 설정 (심한 복합 오차)
    true_acc_bias = np.array([0.2, -0.3, 0.15])
    true_gyr_bias = np.array([0.02, 0.05, -0.02])

    # Scale & Misalignment (꽤 큰 왜곡)
    true_T_acc = np.array(
        [
            [1.10, 0.05, 0.01],  # 10% 스케일 오차
            [0.02, 1.05, 0.02],
            [0.01, 0.01, 0.95],
        ]
    )

    vehicle = GroundVehicle()
    imu = ImuSensor(
        accel_noise=0.01,
        gyro_noise=0.002,  # 노이즈가 적당해야 최적화가 잘됨
        accel_bias=true_acc_bias,
        gyro_bias=true_gyr_bias,
        accel_error_matrix=true_T_acc,
    )

    # 2. 데이터 생성 (Stop & Go)
    print("1. Generating Stop & Go Data...")
    raw_meas, gt_meas = generate_stop_and_go_data(dt, duration, vehicle, imu)

    # 3. SysID Calibration 수행
    print("2. Running SysID Optimization...")
    sysid = SysIdCalibrator()
    opt_params = sysid.run(gt_meas, raw_meas)

    print("\n[Result: Accel Bias]")
    print(f"  True : {true_acc_bias}")
    print(f"  Est  : {opt_params['acc_b']}")
    print(f"  Error: {np.linalg.norm(true_acc_bias - opt_params['acc_b']):.4f}")

    # 4. 항법 비교 (Conventional vs SysID)
    print("\n3. Comparing Navigation Performance...")

    # Conventional: Scale 무시, Bias만 GTSAM으로 (실패할 가능성 높음)
    init_bias = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
    # SysID가 Scale을 잡았으니 Bias는 거의 0일 것임 -> Navigation 시 Zero Bias 사용

    # A. Conventional (Raw Data)
    # 여기선 공정함을 위해 OfflineCalibrator로 Bias라도 잡아봄
    offline = OfflineCalibrator()
    try:
        conv_bias = offline.run(vehicle.poses, raw_meas, dt)
    except:
        conv_bias = init_bias  # 수렴 실패 시

    conv_poses = run_navigation(vehicle.poses[0], raw_meas, conv_bias, dt)

    # B. SysID (Corrected Data)
    # 보정 후 남은 Bias는 거의 없다고 가정 (SysID가 Bias도 포함해서 풀었으므로)
    sysid_poses = run_navigation(vehicle.poses[0], raw_meas, init_bias, dt, params=opt_params)

    # 5. 시각화
    print("4. Visualizing...")

    # Trajectory
    plt.figure(figsize=(10, 6))
    gt_pos = np.array([[p.x(), p.y()] for p in vehicle.poses])
    conv_pos = np.array([[p.x(), p.y()] for p in conv_poses])
    sysid_pos = np.array([[p.x(), p.y()] for p in sysid_poses])

    plt.plot(gt_pos[:, 0], gt_pos[:, 1], "k--", linewidth=2, label="GT")
    plt.plot(conv_pos[:, 0], conv_pos[:, 1], "r-", alpha=0.6, label="Conventional")
    plt.plot(sysid_pos[:, 0], sysid_pos[:, 1], "b-", linewidth=2, label="SysID (Optimization)")
    plt.title("Stop & Go Trajectory Comparison")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
