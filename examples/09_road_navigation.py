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


def run_dead_reckoning(start_pose, start_vel, measurements, bias, dt):
    """Dead Reckoning으로 궤적 복원"""
    pim_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
    pim = gtsam.PreintegratedImuMeasurements(pim_params, bias)

    poses = [start_pose]
    curr_pose = start_pose
    curr_vel = start_vel

    for acc, gyr in measurements:
        pim.integrateMeasurement(acc, gyr, dt)
        nav_state = gtsam.NavState(curr_pose, curr_vel)
        next_state = pim.predict(nav_state, bias)

        curr_pose = next_state.pose()
        curr_vel = next_state.velocity()
        poses.append(curr_pose)
        pim.resetIntegration()

    return poses


def main():
    print("=== Real Road Trajectory Simulation ===")

    # 1. 도로망 궤적 생성
    # 부산 시청 근처
    road_gen = RoadTrajectoryGenerator(location_point=(35.1796, 129.0756), dist=1500)

    print("Finding a route...")
    # 임의의 경로 생성
    x_pts, y_pts, route_ids = road_gen.generate_path()

    # 보간하여 운동학적 데이터 생성 (속도 15m/s = 약 54km/h)
    dt = 0.1
    traj_data = road_gen.interpolate_trajectory(x_pts, y_pts, target_speed=15.0, dt=dt)
    print(f"Trajectory generated: {len(traj_data)} steps ({len(traj_data) * dt:.1f} sec)")

    # 2. IMU 시뮬레이션
    true_acc_bias = np.array([0.1, -0.1, 0.05])
    true_gyr_bias = np.array([0.01, 0.01, -0.01])

    imu = ImuSensor(
        accel_noise=0.05, gyro_noise=0.005, accel_bias=true_acc_bias, gyro_bias=true_gyr_bias
    )

    imu_measurements = []
    gt_poses = []

    for data in traj_data:
        pose = data["pose"]
        acc_body = data["accel_body"]
        omega_body = data["omega_body"]

        meas = imu.measure(pose, acc_body, omega_body)
        imu_measurements.append(meas)
        gt_poses.append(pose)

    # 3. Calibration & Navigation
    print("Running Calibration & Navigation...")

    # 초기 Bias 0 가정
    init_bias = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
    calibrator = OfflineCalibrator(init_bias=init_bias)

    # 최적화 수행
    est_bias = calibrator.run(gt_poses, imu_measurements, dt)
    print(f"Estimated Bias: {est_bias}")

    # 항법 수행 (Dead Reckoning)
    start_vel = gtsam.Point3(*traj_data[0]["vel_body"])

    # 정확히는 World Velocity가 필요
    R0 = gt_poses[0].rotation().matrix()
    start_vel_world = gtsam.Point3(R0 @ traj_data[0]["vel_body"])

    est_poses = run_dead_reckoning(gt_poses[0], start_vel_world, imu_measurements, est_bias, dt)

    # 4. 시각화 (Folium Map)
    print("Generating Map Visualization...")

    # 도로망 원본 경로 (Lat/Lon)
    route_latlon = road_gen.get_latlon_route(route_ids)

    # 중심점 (부산 시청)
    center_lat, center_lon = 35.1796, 129.0756
    m_per_deg_lat = 111000.0
    m_per_deg_lon = 111000.0 * np.cos(np.radians(center_lat))

    # [수정] 시작점 Lat/Lon 가져오기 (G_proj['lat'] 대신 G['y'] 사용)
    start_lat = road_gen.G.nodes[route_ids[0]]["y"]
    start_lon = road_gen.G.nodes[route_ids[0]]["x"]

    est_path_latlon = []
    for p in est_poses:
        # Local meters -> Delta Lat/Lon
        d_lat = p.y() / m_per_deg_lat
        d_lon = p.x() / m_per_deg_lon
        est_path_latlon.append([start_lat + d_lat, start_lon + d_lon])

    # 지도 생성
    m = folium.Map(location=[start_lat, start_lon], zoom_start=15)

    # A. 원본 도로 경로 (빨간색)
    folium.PolyLine(
        locations=route_latlon, color="red", weight=5, opacity=0.5, tooltip="Road Network Path"
    ).add_to(m)

    # B. 추정된 궤적 (파란색)
    folium.PolyLine(
        locations=est_path_latlon,
        color="blue",
        weight=3,
        opacity=0.8,
        tooltip="Estimated Trajectory",
    ).add_to(m)

    # 저장
    map_file = "road_navigation_result.html"
    m.save(map_file)
    print(f"Map saved to '{map_file}'.")

    # 추가: XY Plot
    plt.figure()
    plt.plot([p.x() for p in gt_poses], [p.y() for p in gt_poses], "k--", label="GT (Interpolated)")
    plt.plot([p.x() for p in est_poses], [p.y() for p in est_poses], "b-", label="Estimated")
    plt.plot(x_pts, y_pts, "ro", label="Road Nodes")  # 원본 노드
    plt.legend()
    plt.title("Local Frame Trajectory")
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    main()
