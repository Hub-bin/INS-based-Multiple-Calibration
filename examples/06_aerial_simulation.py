import sys
import os

# 부모 디렉토리를 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from src.dynamics.aerial import AerialVehicle
from src.sensors.imu import ImuSensor


def main():
    print("--- Aerial Vehicle Simulation (Spiral Flight) ---")

    dt = 0.01
    sim_duration = 20.0

    # 1. 드론 생성
    drone = AerialVehicle()

    # IMU 장착
    imu = ImuSensor(accel_noise=0.01, gyro_noise=0.001)

    # 2. 비행 시나리오: 나선형 상승 (Spiral Up)
    vel_x = 5.0
    vel_z = 1.0
    yaw_rate = 0.5

    print("Simulating flight...")

    steps = int(sim_duration / dt)
    timestamps = []
    poses = []

    for t_step in range(steps):
        t = t_step * dt

        velocity_body = np.array([vel_x, 0.0, vel_z])
        omega_body = np.array([0.1 * np.sin(t), 0.1 * np.cos(t), yaw_rate])

        drone.update(dt, velocity_body, omega_body)

        # IMU 측정 (등속 운동 가정, 중력은 내부에서 계산됨)
        meas = imu.measure(
            drone.current_pose, true_accel_body=np.array([0, 0, 0]), true_omega_body=omega_body
        )

        timestamps.append(t)
        poses.append(drone.current_pose)

    # 3. 결과 시각화
    print("Plotting trajectory...")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    xs = [p.x() for p in poses]
    ys = [p.y() for p in poses]
    zs = [p.z() for p in poses]

    ax.plot(xs, ys, zs, label="Drone Trajectory", linewidth=2, color="purple")

    ax.scatter([xs[0]], [ys[0]], [zs[0]], color="green", marker="o", s=50, label="Start")
    ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], color="red", marker="x", s=50, label="End")

    ax.set_title("3D Spiral Flight Simulation")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    ax.grid(True)

    # 비율 맞추기
    max_range = np.array([max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)]).max() / 2.0
    mid_x = (max(xs) + min(xs)) * 0.5
    mid_y = (max(ys) + min(ys)) * 0.5
    mid_z = (max(zs) + min(zs)) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()


if __name__ == "__main__":
    main()
