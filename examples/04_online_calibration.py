import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np

from src.dynamics.ground import GroundVehicle
from src.sensors.imu import ImuSensor
from src.calibration.online import OnlineCalibrator


def main():
    dt = 0.01
    sim_duration = 30.0

    vehicle = GroundVehicle()

    true_accel_bias = [0.2, -0.2, 0.1]
    true_gyro_bias = [0.02, 0.0, -0.01]

    imu = ImuSensor(
        accel_noise=0.01, gyro_noise=0.001, accel_bias=true_accel_bias, gyro_bias=true_gyro_bias
    )

    online_calib = OnlineCalibrator()
    start_vel = np.array([10.0, 0, 0])
    online_calib.initialize(vehicle.current_pose, start_vel)

    history_time = []
    history_est_accel_x = []

    velocity_x = 10.0
    yaw_rate = 0.3
    true_accel_body = np.array([0.0, yaw_rate * velocity_x, 0.0])
    true_omega_body = np.array([0.0, 0.0, yaw_rate])

    print("Running Online Calibration...")
    steps = int(sim_duration / dt)

    for i in range(steps):
        vehicle.update(dt, velocity_x, yaw_rate)
        meas_accel, meas_gyro = imu.measure(vehicle.current_pose, true_accel_body, true_omega_body)

        est_bias = online_calib.update(vehicle.current_pose, meas_accel, meas_gyro, dt)

        history_time.append(vehicle.time)
        history_est_accel_x.append(est_bias.accelerometer()[0])

        if i % 500 == 0:
            print(f"Step {i}/{steps}: Est Bias X = {est_bias.accelerometer()[0]:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(history_time, history_est_accel_x, label="Estimated Bias (Online)")
    plt.axhline(y=true_accel_bias[0], color="r", linestyle="--", label="True Bias")
    plt.title("Online Calibration Convergence")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
