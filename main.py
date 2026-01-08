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
from src.sensors.imu import ImuSensor
from src.calibration.sysid_corrector import SysIdCalibrator

# --- 0. Output Directory ---
OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# --- 1. Helper: Traffic Profile ---
def apply_traffic_profile(trajectory, dt, total_duration_min=10):
    total_steps = int(total_duration_min * 60 / dt)
    n_points = len(trajectory)
    new_traj = []
    curr_idx = 0
    curr_vel = 0.0

    print(f"Generating Traffic Profile for {total_duration_min} mins...")
    for i in range(total_steps):
        t = i * dt
        traffic_phase = int(t / 60.0) % 5
        if traffic_phase == 0 or traffic_phase == 2:
            target_vel = 12.0 + 10.0 * np.sin(2.0 * np.pi * t / 12.0)
            if target_vel < 0:
                target_vel = 0
        elif traffic_phase == 4:
            target_vel = 0.0
        else:
            target_vel = 20.0 + np.random.normal(0, 0.5)

        acc = target_vel - curr_vel
        acc = np.clip(acc, -4.0, 3.5)
        curr_vel += acc * dt
        if curr_vel < 0:
            curr_vel = 0

        step_dist = curr_vel * dt
        curr_idx += step_dist / (2.0)
        idx_int = int(min(curr_idx, n_points - 1))

        new_traj.append(
            {
                "pose": trajectory[idx_int]["pose"],
                "accel_body": np.array([acc, 0.0, 0.0]),
                "omega_body": trajectory[idx_int]["omega_body"] * (curr_vel / 20.0),
                "vel_world": curr_vel,
                "temp": 20.0 + (10.0 * (i / total_steps)),
            }
        )
    return new_traj


# --- 2. Visualization Logic ---
def plot_and_save_all(log, true_params, sim_traj, poses_raw, poses_online, poses_valid, dt):
    times = np.array(log["time"]) / 60.0

    def get_xy(plist):
        out = []
        for p in plist:
            t = p.translation()
            # 안전하게 x, y 추출
            if hasattr(t, "x"):
                out.append([t.x(), t.y()])
            else:
                out.append([t[0], t[1]])
        return np.array(out)

    gt_xy = get_xy([d["pose"] for d in sim_traj])
    xy_raw = get_xy(poses_raw)
    xy_val = get_xy(poses_valid)

    # 데이터 길이 맞춤
    min_l = min(len(gt_xy), len(xy_raw), len(xy_val))
    t_ax = np.arange(min_l) * dt / 60.0

    plt.figure(figsize=(10, 5))
    plt.plot(
        t_ax, np.linalg.norm(gt_xy[:min_l] - xy_raw[:min_l], axis=1), "r--", label="Raw (No Calib)"
    )
    plt.plot(
        t_ax,
        np.linalg.norm(gt_xy[:min_l] - xy_val[:min_l], axis=1),
        "m-",
        label="Proposed (Best Params)",
        lw=2,
    )
    plt.title("Navigation Accuracy Improvement")
    plt.xlabel("Time (min)")
    plt.ylabel("Error (m)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/final_nav_error.png")

    est_b = np.array(log["bias"])
    est_b_plot = est_b.copy()
    est_b_plot[:, 2] += 9.81

    est_h = np.array(log["hyst"])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, (data, true_v, label) in enumerate(
        [(est_b_plot, true_params["bias"], "Bias"), (est_h, true_params["hyst"], "Hyst")]
    ):
        for c in range(3):
            axes[i, c].plot(times, data[:, c], "b", label="Est")
            axes[i, c].axhline(true_v[c], color="r", ls="--", label="True")
            axes[i, c].set_title(f"{label} {'XYZ'[c]}")
            axes[i, c].grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/params_convergence.png")


# --- 3. Main Navigation Logic ---
def navigate(sim_traj, imu_config, mode="online", fixed_params=None, true_params_debug=None):
    print(f"\n>>> Mode: {mode.upper()}")
    imu = ImuSensor(**imu_config)
    sysid = SysIdCalibrator()
    curr_pose = sim_traj[0]["pose"]
    curr_vel = np.zeros(3)
    pim = gtsam.PreintegratedImuMeasurements(
        gtsam.PreintegrationParams.MakeSharedU(0.0), gtsam.imuBias.ConstantBias()
    )

    poses, h_meas, h_true, h_temp = [curr_pose], [], [], []
    param_log = {"time": [], "bias": [], "hyst": []}
    dt = 0.1
    buffer_window = 100
    curr_p = fixed_params
    best_p = None
    min_err = float("inf")

    for i, data in enumerate(sim_traj):
        meas_acc, meas_gyr, _ = imu.measure(
            data["pose"], data["accel_body"], data["omega_body"], temperature=data["temp"]
        )

        if data["vel_world"] < 0.05:
            curr_vel = np.zeros(3)  # ZUPT

        if mode == "online":
            h_meas.append((meas_acc, meas_gyr))

            # [중력 벡터 안전 변환]
            g_body_vec = data["pose"].rotation().unrotate(gtsam.Point3(0, 0, -9.81))
            if isinstance(g_body_vec, np.ndarray):
                g_body = g_body_vec
            else:
                g_body = np.array([g_body_vec.x(), g_body_vec.y(), g_body_vec.z()])

            h_true.append((data["accel_body"], data["omega_body"]))
            h_temp.append(data["temp"])

            if i > 600 and i % buffer_window == 0:
                acc_mask = np.zeros(21)
                acc_mask[9:12] = 1.0
                acc_mask[18:21] = 1.0
                res = sysid.run(h_true, h_meas, h_temp, acc_mask=acc_mask)

                est_b_pure = res["acc_b"].copy()
                est_b_pure[2] += 9.81

                b_err = np.linalg.norm(est_b_pure - true_params_debug["bias"])
                if b_err < min_err:
                    min_err = b_err
                    best_p = copy.deepcopy(res)
                if i % 600 == 0:
                    print(
                        f"  [Time {i * dt / 60:.1f}m] BiasErr: {b_err:.4f} | HystEst: {res['acc_h'][0]:.4f}"
                    )
                curr_p = res

        if i % buffer_window == 0:
            param_log["time"].append(i * dt)
            p = (
                curr_p
                if curr_p
                else {"acc_b": np.zeros(3), "acc_h": np.zeros(3), "acc_T_inv": np.eye(3)}
            )
            param_log["bias"].append(p["acc_b"])
            param_log["hyst"].append(p["acc_h"])

        # [보정 로직]
        if mode == "raw" or curr_p is None:
            # 중력 벡터 안전 변환 (Raw 모드)
            g_body_vec = data["pose"].rotation().unrotate(gtsam.Point3(0, 0, -9.81))
            if isinstance(g_body_vec, np.ndarray):
                g_body = g_body_vec
            else:
                g_body = np.array([g_body_vec.x(), g_body_vec.y(), g_body_vec.z()])

            corr_acc = meas_acc + g_body
        else:
            p = curr_p
            true_diff = np.diff(
                np.array([d["accel_body"][0] for d in sim_traj[max(0, i - 1) : i + 1]]),
                prepend=data["accel_body"][0],
            )
            h_sign = np.tanh(true_diff[-1] * 100.0)

            corr_acc = p["acc_T_inv"] @ (meas_acc + p["acc_b"] - p["acc_h"] * h_sign)

        pim.integrateMeasurement(corr_acc, meas_gyr, dt)
        nav = gtsam.NavState(curr_pose, curr_vel)
        next_s = pim.predict(nav, gtsam.imuBias.ConstantBias())
        curr_pose, curr_vel = next_s.pose(), next_s.velocity()
        pim.resetIntegration()
        poses.append(curr_pose)

    return poses, param_log, (best_p if mode == "online" else curr_p)


def run_main():
    start_loc = (35.1796, 129.0756)
    dt = 0.1
    road_gen = RoadTrajectoryGenerator(location_point=start_loc, dist=8000)
    x, y, _ = road_gen.generate_path()
    sim_traj = apply_traffic_profile(road_gen.interpolate_trajectory(x, y, 20.0, dt), dt)

    true_p = {"bias": np.array([0.05, -0.02, 0.01]), "hyst": np.full(3, 0.05), "T_inv": np.eye(3)}
    imu_cfg = {"accel_bias": true_p["bias"], "accel_hysteresis": 0.05, "accel_noise": 0.0001}

    p_raw, _, _ = navigate(sim_traj, imu_cfg, "raw")
    p_on, log, b_p = navigate(sim_traj, imu_cfg, "online", true_params_debug=true_p)
    p_val, _, _ = navigate(sim_traj, imu_cfg, "fixed", fixed_params=b_p)

    print(f"\n[Final Results]")
    est_b_pure = b_p["acc_b"].copy()
    est_b_pure[2] += 9.81
    print(f"  > Bias (Est/True): {est_b_pure} / {true_p['bias']}")
    print(f"  > Hyst (Est/True): {b_p['acc_h']} / {true_p['hyst']}")

    # Map Save
    m = folium.Map(location=start_loc, zoom_start=14)

    def add_trace(plist, color, name):
        m_deg = 111000.0
        lon_s = m_deg * np.cos(np.radians(start_loc[0]))
        coords = []
        for p in plist:
            t = p.translation()
            # 안전하게 x, y 추출
            if hasattr(t, "x"):
                px, py = t.x(), t.y()
            else:
                px, py = t[0], t[1]
            coords.append([start_loc[0] + py / m_deg, start_loc[1] + px / lon_s])
        # 길이에 맞춰 6000개만 그림
        folium.PolyLine(coords[:6000], color=color, weight=3, tooltip=name).add_to(m)

    add_trace([d["pose"] for d in sim_traj], "green", "GT")
    add_trace(p_raw, "red", "Raw")
    add_trace(p_val, "purple", "Proposed")
    m.save(f"{OUTPUT_DIR}/final_map.html")
    print(f"Done. Check '{OUTPUT_DIR}/'")

    plot_and_save_all(log, true_p, sim_traj, p_raw, p_on, p_val, dt)
    plt.show()


if __name__ == "__main__":
    run_main()
