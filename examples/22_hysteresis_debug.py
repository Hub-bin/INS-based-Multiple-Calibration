import sys
import os
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import gtsam
import folium
import torch
import copy

from src.utils.road_generator import RoadTrajectoryGenerator
from src.dynamics.ground import GroundVehicle
from src.sensors.imu import ImuSensor
from src.calibration.sysid_corrector import SysIdCalibrator
from src.calibration.rl_agent import RLAgent

OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# --- 1. Traffic (Aggressive) ---
def apply_aggressive_traffic(trajectory, dt, total_duration_min=10):
    total_steps = int(total_duration_min * 60 / dt)
    n_points = len(trajectory)
    new_traj = []
    curr_idx = 0
    curr_vel = 0.0
    print(f"Generating Aggressive Traffic Profile ({total_duration_min} mins)...")
    for i in range(total_steps):
        t = i * dt
        phase = int(t / 20.0) % 2
        if phase == 0:
            target_vel = 15.0 + 10.0 * np.sin(2.0 * np.pi * t / 5.0)  # 빠른 진동 (5초 주기)
            if target_vel < 0:
                target_vel = 0
        else:
            target_vel = 20.0

        acc = (target_vel - curr_vel) / dt * 0.2
        acc = np.clip(acc, -5.0, 5.0)
        curr_vel += acc * dt
        if curr_vel < 0:
            curr_vel = 0
        step_dist = curr_vel * dt
        curr_idx += step_dist / (20.0 * 0.1)
        if curr_idx >= n_points - 1:
            curr_idx = n_points - 1
            curr_vel = 0.0

        idx_int = int(curr_idx)
        alpha = curr_idx - idx_int
        p1 = trajectory[idx_int]["pose"]
        p2 = trajectory[min(idx_int + 1, n_points - 1)]["pose"]
        t1 = p1.translation()
        t2 = p2.translation()
        if not isinstance(t1, np.ndarray):
            t1 = np.array([t1.x(), t1.y(), t1.z()])
        if not isinstance(t2, np.ndarray):
            t2 = np.array([t2.x(), t2.y(), t2.z()])
        t_interp = t1 * (1 - alpha) + t2 * alpha
        r1 = p1.rotation()
        r2 = p2.rotation()
        r_interp = r1.slerp(alpha, r2)

        acc_body = np.array([acc, 0.0, 0.0])
        orig_omega = trajectory[idx_int]["omega_body"]
        scaled_omega = orig_omega * (curr_vel / 20.0)
        new_traj.append(
            {
                "pose": gtsam.Pose3(r_interp, t_interp),
                "accel_body": acc_body,
                "omega_body": scaled_omega,
                "vel_world": curr_vel,
                "temp": 20.0,
            }
        )
    return new_traj


# --- 2. Scatter Plot for Debugging ---
def plot_hysteresis_correlation(true_data, meas_data, bias):
    # Calculate Residual (Meas - True - Bias)
    # This residual should ideally be "Hysteresis + Noise"
    true_acc = np.array([t[0] for t in true_data])
    meas_acc = np.array([m[0] for m in meas_data])

    residual = meas_acc - true_acc - bias

    # Calculate Feature (tanh(diff))
    diff = np.diff(true_acc, axis=0, prepend=true_acc[0:1])
    feature = np.tanh(diff * 100.0)

    # Scatter Plot (Only X-axis for clarity)
    plt.figure(figsize=(8, 8))
    plt.scatter(feature[:, 0], residual[:, 0], alpha=0.1, s=1, color="blue", label="Data Points")

    # Linear Fit (Slope should be Hysteresis)
    # y = ax + b
    A = np.vstack([feature[:, 0], np.ones(len(feature))]).T
    m, c = np.linalg.lstsq(A, residual[:, 0], rcond=None)[0]

    plt.plot(
        feature[:, 0],
        m * feature[:, 0] + c,
        "r",
        linewidth=3,
        label=f"Fit Slope (Est Hyst) = {m:.4f}",
    )

    plt.title("[Debug] Hysteresis Correlation (Residual vs Change Direction)")
    plt.xlabel("Feature: tanh(Change)")
    plt.ylabel("Residual: Meas - True - Bias")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/figure_4_hyst_correlation.png")
    # plt.show()


# --- 3. Navigation ---
def run_debug():
    print("=== Hysteresis Correlation Check ===")
    start_loc = (35.1796, 129.0756)
    dt = 0.1
    road_gen = RoadTrajectoryGenerator(location_point=start_loc, dist=8000)
    x_pts, y_pts, _ = road_gen.generate_path()
    base_traj = road_gen.interpolate_trajectory(x_pts, y_pts, target_speed=20.0, dt=dt)
    sim_traj = apply_aggressive_traffic(base_traj, dt, total_duration_min=5)  # 5분만

    # Only Bias and Hysteresis (Remove Temp for clear signal)
    true_params = {
        "bias": np.array([0.05, 0.0, 0.0]),
        "hyst": np.array([0.05, 0.0, 0.0]),  # Target: 0.05
    }

    imu_config = {
        "accel_bias": true_params["bias"],
        "accel_hysteresis": 0.05,
        "accel_noise": 0.0001,  # Low noise for clear plot
        "gyro_noise": 0.00001,
    }

    imu = ImuSensor(**imu_config)
    sysid = SysIdCalibrator()

    history_meas, history_true, history_temp = [], [], []

    print("Collecting Data...")
    for i, data in enumerate(sim_traj):
        meas_acc, meas_gyr, _ = imu.measure(
            data["pose"], data["accel_body"], data["omega_body"], temperature=data["temp"]
        )

        history_meas.append((meas_acc, meas_gyr))
        rot = data["pose"].rotation()
        g_body_np = rot.unrotate(gtsam.Point3(0, 0, -9.81))
        if not isinstance(g_body_np, np.ndarray):
            g_body_np = np.array([g_body_np.x(), g_body_np.y(), g_body_np.z()])
        sf_true = data["accel_body"] - g_body_np
        history_true.append((sf_true, data["omega_body"]))
        history_temp.append(data["temp"])

    print("Running SysID...")
    # Bias Mask + Hyst Mask
    acc_mask = np.zeros(21)
    acc_mask[9:12] = 1.0
    acc_mask[18:21] = 1.0

    res = sysid.run(history_true, history_meas, history_temp, acc_mask=acc_mask)

    print(f"\n[Result]")
    print(f"  > Est Bias: {res['acc_b'][0]:.4f} (True: 0.05)")
    print(f"  > Est Hyst: {res['acc_h'][0]:.4f} (True: 0.05)")

    # Correlation Plot
    plot_hysteresis_correlation(history_true, history_meas, res["acc_b"])

    # Save Results
    times = np.arange(len(history_meas)) * dt / 60.0
    plt.figure()
    plt.plot(times, [res["acc_h"][0]] * len(times), "b-", label="Final Est")
    plt.axhline(0.05, color="r", ls="--", label="True")
    plt.title("Hysteresis Est")
    plt.savefig(f"{OUTPUT_DIR}/figure_2_hysteresis.png")

    print(f"\nCheck '{OUTPUT_DIR}/figure_4_hyst_correlation.png' to see the signal!")


if __name__ == "__main__":
    run_debug()
