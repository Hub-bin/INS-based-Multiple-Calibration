import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import gtsam

from src.dynamics.ground import GroundVehicle
from src.sensors.camera import CameraSensor


def generate_landmarks(x_range, y_range, z_range, count):
    landmarks = {}
    for i in range(count):
        x = np.random.uniform(*x_range)
        y = np.random.uniform(*y_range)
        z = np.random.uniform(*z_range)
        landmarks[i] = gtsam.Point3(x, y, z)
    return landmarks


def main():
    dt = 0.1
    sim_duration = 10.0

    vehicle = GroundVehicle()
    camera = CameraSensor(noise_sigma=0.0)

    landmarks = generate_landmarks((-20, 120), (-50, 100), (0, 5), count=50)

    velocity_x = 10.0
    yaw_rate = 0.2

    history_poses = []

    print("Running Camera Simulation...")
    steps = int(sim_duration / dt)

    plt.figure(figsize=(10, 8))

    # 랜드마크 (인덱싱 사용)
    lx = [p[0] for p in landmarks.values()]
    ly = [p[1] for p in landmarks.values()]
    plt.scatter(lx, ly, c="gray", marker="x", label="Landmarks")

    for i in range(steps):
        vehicle.update(dt, velocity_x, yaw_rate)

        if i % 10 == 0:
            observations, cam_pose = camera.measure(vehicle.current_pose, landmarks)
            history_poses.append((vehicle.current_pose.x(), vehicle.current_pose.y()))

            cam_x = cam_pose.x()
            cam_y = cam_pose.y()

            for l_id, uv in observations:
                l_pos = landmarks[l_id]
                # 카메라 -> 랜드마크 연결선 (인덱싱 사용)
                plt.plot([cam_x, l_pos[0]], [cam_y, l_pos[1]], "r-", alpha=0.3, linewidth=0.5)

    traj_x = [p[0] for p in history_poses]
    traj_y = [p[1] for p in history_poses]
    plt.plot(traj_x, traj_y, "b-", linewidth=2, label="Vehicle Trajectory")

    plt.title("Camera FOV Visualization")
    plt.axis("equal")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
