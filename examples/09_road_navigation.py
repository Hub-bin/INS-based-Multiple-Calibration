import sys
import os

# 부모 디렉토리 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import gtsam
import folium

from src.utils.road_generator import RoadTrajectoryGenerator
from src.sensors.imu import ImuSensor
from src.calibration.offline import OfflineCalibrator
from src.calibration.sysid_corrector import SysIdCalibrator  # [New] SysID 추가


def run_dead_reckoning(start_pose, start_vel, measurements, bias, dt, correction_params=None):
    """Dead Reckoning으로 궤적 복원 (SysID 보정 적용)"""
    pim_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
    pim = gtsam.PreintegratedImuMeasurements(pim_params, bias)

    poses = [start_pose]
    curr_pose = start_pose
    curr_vel = start_vel

    # SysID 파라미터 추출
    if correction_params:
        acc_T_inv = correction_params.get("acc_T_inv", np.eye(3))
        acc_b = correction_params.get("acc_b", np.zeros(3))
        gyr_T_inv = correction_params.get("gyr_T_inv", np.eye(3))
        gyr_b = correction_params.get("gyr_b", np.zeros(3))

    for raw_acc, raw_gyr in measurements:
        if correction_params:
            # SysID Correction: T_inv * (Raw - b)
            acc = acc_T_inv @ (raw_acc - acc_b)
            gyr = gyr_T_inv @ (raw_gyr - gyr_b)
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
    print("=== Real Road Trajectory Simulation (Bias + Scale + Misalignment) ===")

    # 1. 도로망 궤적 생성
    # 부산 시청 근처
    road_gen = RoadTrajectoryGenerator(location_point=(35.1796, 129.0756), dist=1500)

    print("Finding a route...")
    x_pts, y_pts, route_ids = road_gen.generate_path()

    # 약 20m/s (72km/h)로 설정하여 다이나믹한 움직임 유도
    dt = 0.1
    traj_data = road_gen.interpolate_trajectory(x_pts, y_pts, target_speed=20.0, dt=dt)
    print(f"Trajectory generated: {len(traj_data)} steps ({len(traj_data) * dt:.1f} sec)")

    # 2. IMU 시뮬레이션 (복합 오차 주입)
    true_acc_bias = np.array([0.2, -0.2, 0.1])
    true_gyr_bias = np.array([0.01, 0.02, -0.01])

    # Scale Factor (대각선) & Misalignment (비대각선)
    true_T_acc = np.array([[1.05, 0.02, 0.01], [0.02, 1.03, 0.01], [0.01, 0.01, 0.98]])
    true_T_gyr = np.eye(3)

    imu = ImuSensor(
        accel_noise=0.02,
        gyro_noise=0.005,
        accel_bias=true_acc_bias,
        gyro_bias=true_gyr_bias,
        accel_error_matrix=true_T_acc,
        gyro_error_matrix=true_T_gyr,
    )

    imu_measurements = []
    gt_measurements = []
    gt_poses = []

    # 초기 속도 (World Frame)
    if len(traj_data) > 0:
        R0 = traj_data[0]["pose"].rotation().matrix()
        start_vel_world = gtsam.Point3(R0 @ traj_data[0]["vel_body"])
    else:
        print("Error: No trajectory data generated.")
        return

    for data in traj_data:
        pose = data["pose"]
        acc_kinematic_body = data["accel_body"]
        omega_body = data["omega_body"]

        # 1. 센서 측정 (Raw)
        meas = imu.measure(pose, acc_kinematic_body, omega_body)
        imu_measurements.append(meas)
        gt_poses.append(pose)

        # 2. SysID용 정답 데이터 생성 (Specific Force)
        rot_wb = pose.rotation()
        g_world = gtsam.Point3(0, 0, -9.81)
        g_body_gtsam = rot_wb.unrotate(g_world)

        # [수정] numpy array 반환 대응
        if isinstance(g_body_gtsam, np.ndarray):
            g_body = g_body_gtsam
        else:
            g_body = np.array([g_body_gtsam.x(), g_body_gtsam.y(), g_body_gtsam.z()])

        sf_body = acc_kinematic_body - g_body
        gt_measurements.append((sf_body, omega_body))

    # 3. Calibration (SysID)
    print("Running SysID Calibration...")
    sysid = SysIdCalibrator()
    correction_params = sysid.run(gt_measurements, imu_measurements)

    print("Estimated Accel Bias:", correction_params["acc_b"])
    print("True Accel Bias:     ", true_acc_bias)

    # 4. 항법 수행 (Dead Reckoning)
    zero_bias = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))

    est_poses = run_dead_reckoning(
        gt_poses[0],
        start_vel_world,
        imu_measurements,
        zero_bias,
        dt,
        correction_params=correction_params,
    )

    # 5. 시각화 (Folium Map)
    print("Generating Map Visualization...")
    route_latlon = road_gen.get_latlon_route(route_ids)

    # 시작점 Lat/Lon
    start_lat = road_gen.G.nodes[route_ids[0]]["y"]
    start_lon = road_gen.G.nodes[route_ids[0]]["x"]

    # 좌표 변환 계수
    m_per_deg_lat = 111000.0
    m_per_deg_lon = 111000.0 * np.cos(np.radians(start_lat))

    est_path_latlon = []
    for p in est_poses:
        d_lat = p.y() / m_per_deg_lat
        d_lon = p.x() / m_per_deg_lon
        est_path_latlon.append([start_lat + d_lat, start_lon + d_lon])

    m = folium.Map(location=[start_lat, start_lon], zoom_start=15)

    # 원본 도로 (Red)
    folium.PolyLine(
        locations=route_latlon, color="red", weight=5, opacity=0.5, tooltip="Road Network"
    ).add_to(m)

    # 추정 궤적 (Blue)
    folium.PolyLine(
        locations=est_path_latlon, color="blue", weight=3, opacity=0.8, tooltip="Estimated (SysID)"
    ).add_to(m)

    map_file = "road_navigation_result_advanced.html"
    m.save(map_file)
    print(f"Map saved to '{map_file}'.")

    # XY Plot
    gt_x = [p.x() for p in gt_poses]
    gt_y = [p.y() for p in gt_poses]
    est_x = [p.x() for p in est_poses]
    est_y = [p.y() for p in est_poses]

    plt.figure(figsize=(10, 6))
    plt.plot(gt_x, gt_y, "k--", label="GT")
    plt.plot(est_x, est_y, "b-", label="Estimated (SysID)")
    plt.title("Road Navigation with Scale/Misalignment Errors")
    plt.axis("equal")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
