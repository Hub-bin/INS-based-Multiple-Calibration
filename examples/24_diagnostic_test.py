import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import gtsam
import copy

# 기존 모듈 활용
from src.utils.road_generator import RoadTrajectoryGenerator
from src.sensors.imu import ImuSensor
from src.simulation.profile import TrajectorySimulator
from src.navigation.strapdown import StrapdownNavigator

OUTPUT_DIR = "output_diag"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def plot_diagnostic(test_name, log_data, dt):
    time = np.array(log_data["time"])

    # 1. Velocity Analysis
    vel_est = np.array(log_data["vel_est"])
    vel_gt = np.array(log_data["vel_gt"])

    plt.figure(figsize=(12, 10))
    plt.suptitle(f"[{test_name}] Diagnostics", fontsize=16)

    # Velocity Comparison
    for i in range(3):
        plt.subplot(3, 2, i * 2 + 1)
        plt.plot(time, vel_gt[:, i], "k--", label="GT", alpha=0.7)
        plt.plot(time, vel_est[:, i], "r-", label="Est")
        plt.title(f"Velocity {'XYZ'[i]} (m/s)")
        plt.grid(True)
        if i == 0:
            plt.legend()

    # 2. Position Error
    pos_est = np.array(log_data["pos_est"])
    pos_gt = np.array(log_data["pos_gt"])

    # 크기 맞춤 (혹시 모를 off-by-one 방지)
    min_len = min(len(pos_est), len(pos_gt))
    pos_est = pos_est[:min_len]
    pos_gt = pos_gt[:min_len]
    time = time[:min_len]

    pos_err = pos_est - pos_gt

    for i in range(3):
        plt.subplot(3, 2, i * 2 + 2)
        plt.plot(time, pos_err[:, i], "b-")
        plt.title(f"Pos Error {'XYZ'[i]} (m)")
        plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{OUTPUT_DIR}/{test_name}_analysis.png")
    plt.close()

    # 3. IMU Input Analysis (Specific Force)
    meas_acc = np.array(log_data["meas_acc"])
    meas_acc = meas_acc[:min_len]

    plt.figure(figsize=(10, 6))
    for i in range(3):
        plt.plot(time, meas_acc[:, i], label=f"Acc {'XYZ'[i]}")
    plt.title(f"[{test_name}] IMU Input (Specific Force)")
    plt.ylabel("m/s^2")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/{test_name}_imu_input.png")
    plt.close()


def run_static_test():
    print("\n>>> [Test 1] Static Test (Duration: 60s)")
    dt = 0.1
    steps = int(60 / dt)

    # Static Data Generation
    start_pose = gtsam.Pose3()

    # Navigator Setup (Standard Gravity 9.81)
    nav = StrapdownNavigator(start_pose, gravity=9.81)

    log = {"time": [], "vel_est": [], "vel_gt": [], "pos_est": [], "pos_gt": [], "meas_acc": []}

    for i in range(steps):
        t = i * dt

        # Static Condition: SF = [0, 0, 9.81] (Z-up local frame)
        sf_true = np.array([0.0, 0.0, 9.81])
        omega_true = np.array([0.0, 0.0, 0.0])

        # Integration
        nav.integrate(sf_true, omega_true, dt)
        pose_est = nav.predict()

        # Logging
        log["time"].append(t)
        log["vel_est"].append(nav.curr_vel)
        log["vel_gt"].append(np.zeros(3))

        # [수정] Pose에서 Translation 추출 시 안전하게 처리
        p = pose_est.translation()
        if hasattr(p, "x"):
            log["pos_est"].append([p.x(), p.y(), p.z()])
        else:
            log["pos_est"].append([p[0], p[1], p[2]])

        log["pos_gt"].append([0, 0, 0])
        log["meas_acc"].append(sf_true)

    # Analyze Final Drift
    final_z_err = log["pos_est"][-1][2]
    final_z_vel = log["vel_est"][-1][2]
    print(f"  > Final Z Position Error: {final_z_err:.4f} m")
    print(f"  > Final Z Velocity Error: {final_z_vel:.4f} m/s")

    if abs(final_z_err) > 1.0:
        print("  [FAIL] Large Drift detected. Gravity cancellation failed.")
    else:
        print("  [PASS] Static test passed.")

    plot_diagnostic("static_test", log, dt)


def run_zero_error_test():
    print("\n>>> [Test 2] Zero-Error Dynamic Test (Duration: 60s)")
    dt = 0.1

    # 1. Generate Trajectory
    start_loc = (35.1796, 129.0756)
    road_gen = RoadTrajectoryGenerator(location_point=start_loc, dist=2000)
    sim = TrajectorySimulator(road_gen, dt)
    traj_data = sim.generate_3d_profile(total_duration_min=1.0)

    # 2. Perfect Sensor (No Noise, No Bias)
    imu = ImuSensor(
        accel_bias=[0, 0, 0],
        accel_hysteresis=[0, 0, 0],
        accel_noise=0.0,
        gyro_bias=[0, 0, 0],
        gyro_noise=0.0,
    )

    # 3. Navigator
    start_pose = traj_data[0]["pose"]
    nav = StrapdownNavigator(start_pose, gravity=9.81)
    nav.curr_vel = traj_data[0]["vel_world"]  # Init Velocity

    log = {"time": [], "vel_est": [], "vel_gt": [], "pos_est": [], "pos_gt": [], "meas_acc": []}

    for i, data in enumerate(traj_data):
        # Measure (Perfect Sensor)
        meas_acc, meas_gyr, _ = imu.measure(
            data["pose"], data["sf_true"], data["omega_body"], data["temp"]
        )

        # Integration
        nav.integrate(meas_acc, meas_gyr, dt)
        pose_est = nav.predict()

        # Logging
        log["time"].append(data["time"])
        log["vel_est"].append(nav.curr_vel)
        log["vel_gt"].append(data["vel_world"])

        # [수정] Est Pose 안전 추출
        pe = pose_est.translation()
        if hasattr(pe, "x"):
            log["pos_est"].append([pe.x(), pe.y(), pe.z()])
        else:
            log["pos_est"].append([pe[0], pe[1], pe[2]])

        # [수정] GT Pose 안전 추출
        pg = data["pose"].translation()
        if hasattr(pg, "x"):
            log["pos_gt"].append([pg.x(), pg.y(), pg.z()])
        else:
            log["pos_gt"].append([pg[0], pg[1], pg[2]])

        log["meas_acc"].append(meas_acc)

    # Analyze
    pos_est_final = np.array(log["pos_est"][-1])
    pos_gt_final = np.array(log["pos_gt"][-1])
    final_err = np.linalg.norm(pos_est_final - pos_gt_final)

    print(f"  > Final Position Error: {final_err:.4f} m")

    if final_err > 10.0:
        print("  [FAIL] Dynamic tracking failed even with perfect sensors.")
    else:
        print("  [PASS] Zero-error dynamic test passed.")

    plot_diagnostic("zero_error_test", log, dt)


if __name__ == "__main__":
    run_static_test()
    run_zero_error_test()
