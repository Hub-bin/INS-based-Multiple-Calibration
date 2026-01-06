import sys
import os

# 부모 디렉토리를 경로에 추가하여 src 모듈을 찾을 수 있게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import gtsam

from src.dynamics.ground import GroundVehicle
from src.sensors.imu import ImuSensor


def plot_results(poses, imu_data):
    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121, projection="3d")
    x_vals = [p.x() for p in poses]
    y_vals = [p.y() for p in poses]
    z_vals = [p.z() for p in poses]
    ax1.plot(x_vals, y_vals, z_vals, label="Ground Truth", linewidth=2)
    ax1.set_title("Vehicle Trajectory")
    ax1.legend()

    ax2 = fig.add_subplot(222)
    accel_x = [d[0][0] for d in imu_data]
    accel_y = [d[0][1] for d in imu_data]
    accel_z = [d[0][2] for d in imu_data]
    ax2.plot(accel_x, label="Accel X", alpha=0.5)
    ax2.plot(accel_y, label="Accel Y", alpha=0.5)
    ax2.plot(accel_z, label="Accel Z", alpha=0.5)
    ax2.set_title("IMU Accelerometer")
    ax2.legend()
    ax2.grid(True)

    ax3 = fig.add_subplot(224)
    gyro_x = [d[1][0] for d in imu_data]
    gyro_y = [d[1][1] for d in imu_data]
    gyro_z = [d[1][2] for d in imu_data]
    ax3.plot(gyro_x, label="Gyro X", alpha=0.5)
    ax3.plot(gyro_y, label="Gyro Y", alpha=0.5)
    ax3.plot(gyro_z, label="Gyro Z", alpha=0.5)
    ax3.set_title("IMU Gyroscope")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    dt = 0.01
    sim_duration = 15.0
    vehicle = GroundVehicle()

    imu = ImuSensor(
        accel_noise=0.1,
        gyro_noise=0.01,
        accel_bias=[0.1, -0.1, 0.05],
        gyro_bias=[0.001, 0.001, 0.001],
    )

    velocity_x = 10.0
    yaw_rate = 0.5

    true_accel_body = np.array([0.0, yaw_rate * velocity_x, 0.0])
    true_omega_body = np.array([0.0, 0.0, yaw_rate])

    imu_measurements = []
    steps = int(sim_duration / dt)

    print("Running IMU Simulation...")
    for _ in range(steps):
        vehicle.update(dt, velocity_x, yaw_rate)
        meas = imu.measure(vehicle.current_pose, true_accel_body, true_omega_body)
        imu_measurements.append(meas)

    plot_results(vehicle.poses[1:], imu_measurements)


if __name__ == "__main__":
    main()
