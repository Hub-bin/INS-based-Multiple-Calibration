import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import gtsam

from src.dynamics.ground import GroundVehicle
from src.sensors.lidar import LidarSensor


def generate_random_landmarks(count=100, area_size=100):
    landmarks = {}
    for i in range(count):
        x = np.random.uniform(-area_size, area_size)
        y = np.random.uniform(-area_size, area_size)
        z = np.random.uniform(0, 3)
        landmarks[i] = gtsam.Point3(x, y, z)
    return landmarks


def main():
    dt = 0.1
    sim_duration = 5.0

    vehicle = GroundVehicle()
    lidar = LidarSensor(max_range=20.0, range_noise=0.1)
    landmarks = generate_random_landmarks(count=300, area_size=50)

    velocity_x = 5.0
    yaw_rate = 0.5

    print("Running LiDAR Simulation...")
    steps = int(sim_duration / dt)
    for _ in range(steps):
        vehicle.update(dt, velocity_x, yaw_rate)

    scan_points, lidar_pose = lidar.measure(vehicle.current_pose, landmarks)

    plt.figure(figsize=(10, 10))

    # 전체 랜드마크 (인덱싱 사용)
    lx = [p[0] for p in landmarks.values()]
    ly = [p[1] for p in landmarks.values()]
    plt.scatter(lx, ly, c="lightgray", marker=".", label="All Landmarks")

    detected_x = []
    detected_y = []
    T_wl = lidar_pose

    for lp in scan_points:
        wp = T_wl.transformFrom(lp)  # wp is numpy array
        detected_x.append(wp[0])
        detected_y.append(wp[1])

    plt.scatter(detected_x, detected_y, c="red", s=20, marker="x", label="LiDAR Detected")

    veh_x = vehicle.current_pose.x()
    veh_y = vehicle.current_pose.y()
    plt.plot(veh_x, veh_y, "bo", markersize=10, label="Vehicle")

    circle = plt.Circle(
        (veh_x, veh_y), lidar.max_range, color="blue", fill=False, linestyle="--", label="Max Range"
    )
    plt.gca().add_patch(circle)

    plt.title(f"LiDAR Simulation (Range: {lidar.max_range}m)")
    plt.axis("equal")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
