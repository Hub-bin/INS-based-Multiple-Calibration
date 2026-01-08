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
from src.calibration.rl_agent import RLAgent

# --- 0. Output Directory ---
OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# --- 1. Helper: Traffic & Road Dynamics ---
def apply_traffic_profile(trajectory, dt, total_duration_min=10):
    total_steps = int(total_duration_min * 60 / dt)
    n_points = len(trajectory)
    new_traj = []
    curr_idx = 0
    curr_vel = 0.0

    print(f"Generating 3D Traffic with Road Dynamics ({total_duration_min} mins)...")
    for i in range(total_steps):
        t = i * dt
        # 1. 속도 프로파일 (Stop & Go for Hysteresis)
        traffic_phase = int(t / 60.0) % 5
        if traffic_phase == 0 or traffic_phase == 2:
            target_vel = 15.0 + 10.0 * np.sin(2.0 * np.pi * t / 15.0)
            if target_vel < 0:
                target_vel = 0
        elif traffic_phase == 4:
            target_vel = 0.0
        else:
            target_vel = 25.0 + np.random.normal(0, 0.5)

        acc_lin = target_vel - curr_vel
        acc_lin = np.clip(acc_lin, -4.0, 3.5)
        curr_vel += acc_lin * dt
        if curr_vel < 0:
            curr_vel = 0

        # 위치 업데이트
        step_dist = curr_vel * dt
        curr_idx += step_dist / (2.0)
        idx_int = int(min(curr_idx, n_points - 1))

        # 2. 자세(Attitude) 생성
        base_pose = trajectory[idx_int]["pose"]
        base_rot = base_pose.rotation()

        # Road Dynamics: Roll(뱅크), Pitch(경사)
        sim_roll = 0.1 * np.sin(2.0 * np.pi * t / 17.0)
        sim_pitch = 0.08 * np.sin(2.0 * np.pi * t / 29.0)

        delta_rot = gtsam.Rot3.Ypr(0, sim_pitch, sim_roll)
        new_rot = base_rot.compose(delta_rot)
        new_pose = gtsam.Pose3(new_rot, base_pose.translation())

        # 3. 3축 가속도/각속도 생성
        acc_x = acc_lin
        acc_y = 0.8 * np.sin(2.0 * np.pi * t / 7.0)
        acc_z = 0.5 * np.sin(2.0 * np.pi * t / 3.0)

        omega = trajectory[idx_int]["omega_body"] * (curr_vel / 20.0)
        omega += np.array(
            [
                0.1 * np.cos(2 * np.pi * t / 17.0),
                0.05 * np.cos(2 * np.pi * t / 29.0),
                0.02 * np.sin(2 * np.pi * t / 10.0),
            ]
        )

        new_traj.append(
            {
                "pose": new_pose,
                "accel_body": np.array([acc_x, acc_y, acc_z]),
                "omega_body": omega,
                "vel_world": curr_vel,
                "temp": 20.0 + (30.0 * (i / total_steps)),
            }
        )
    return new_traj


# --- 2. Visualization: All Parameters ---
def plot_all_calibration_results(log, true_params_acc, true_params_gyr, dt):
    times = np.array(log["time"]) / 60.0

    param_groups = [
        ("Bias", "bias", "bias"),
        ("Scale Factor", "scale", "scale"),
        ("Temp Linear", "temp_lin", "temp_lin"),
        ("Temp Nonlinear", "temp_non", "temp_non"),
        ("Hysteresis", "hyst", "hyst"),
    ]

    # 1. Accelerometer Plots
    fig_acc, axes_acc = plt.subplots(5, 3, figsize=(15, 15))
    fig_acc.suptitle("Accelerometer Calibration Parameters", fontsize=16)

    for r, (name, l_key, t_key) in enumerate(param_groups):
        est_data = np.array(log["acc"][l_key])
        true_data = true_params_acc[t_key]

        for c in range(3):
            ax = axes_acc[r, c]
            ax.plot(times, est_data[:, c], "b-", label="Est")

            tv = true_data[c] if hasattr(true_data, "__len__") else true_data
            if name == "Scale Factor" and not hasattr(true_data, "__len__"):
                tv = 1.0

            ax.axhline(tv, color="r", linestyle="--", label="True")

            if c == 0:
                ax.set_ylabel(name)
            if r == 0:
                ax.set_title(f"{'XYZ'[c]}-axis")
            if r == 4:
                ax.set_xlabel("Time (min)")
            ax.grid(True)
            if r == 0 and c == 2:
                ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{OUTPUT_DIR}/accel_params.png")

    # 2. Gyroscope Plots
    fig_gyr, axes_gyr = plt.subplots(5, 3, figsize=(15, 15))
    fig_gyr.suptitle("Gyroscope Calibration Parameters", fontsize=16)

    for r, (name, l_key, t_key) in enumerate(param_groups):
        est_data = np.array(log["gyr"][l_key])
        true_data = true_params_gyr[t_key]

        for c in range(3):
            ax = axes_gyr[r, c]
            ax.plot(times, est_data[:, c], "g-", label="Est")

            tv = true_data[c] if hasattr(true_data, "__len__") else true_data
            if name == "Scale Factor" and not hasattr(true_data, "__len__"):
                tv = 1.0

            ax.axhline(tv, color="r", linestyle="--", label="True")

            if c == 0:
                ax.set_ylabel(name)
            if r == 0:
                ax.set_title(f"{'XYZ'[c]}-axis")
            if r == 4:
                ax.set_xlabel("Time (min)")
            ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{OUTPUT_DIR}/gyro_params.png")


# --- 3. Navigation Engine ---
def navigate(
    sim_traj, imu_config, mode="online", fixed_params=None, true_p_acc=None, true_p_gyr=None
):
    print(f"\n>>> Mode: {mode.upper()}")
    imu = ImuSensor(**imu_config)
    sysid = SysIdCalibrator()
    curr_pose = sim_traj[0]["pose"]
    curr_vel = np.zeros(3)
    pim = gtsam.PreintegratedImuMeasurements(
        gtsam.PreintegrationParams.MakeSharedU(0.0), gtsam.imuBias.ConstantBias()
    )

    poses, h_meas, h_true_acc, h_true_gyr, h_temp = [curr_pose], [], [], [], []
    param_log = {
        "time": [],
        "acc": {"bias": [], "scale": [], "temp_lin": [], "temp_non": [], "hyst": []},
        "gyr": {"bias": [], "scale": [], "temp_lin": [], "temp_non": [], "hyst": []},
    }
    dt = 0.1
    buffer_window = 100
    curr_p = fixed_params
    best_p = None
    min_err = float("inf")

    prev_sf = np.zeros(3)
    prev_omega = np.zeros(3)

    for i, data in enumerate(sim_traj):
        # 1. Measure
        meas_acc, meas_gyr, _ = imu.measure(
            data["pose"], data["accel_body"], data["omega_body"], temperature=data["temp"]
        )

        # [수정] Scale Factor 수동 적용 (ImuSensor 미지원 기능 보완)
        if true_p_acc and true_p_gyr:
            meas_acc = meas_acc * true_p_acc["scale"]
            meas_gyr = meas_gyr * true_p_gyr["scale"]

        # 2. ZUPT
        if data["vel_world"] < 0.05:
            curr_vel = np.zeros(3)

        # 3. Online Learning
        if mode == "online":
            h_meas.append((meas_acc, meas_gyr))

            g_body_vec = data["pose"].rotation().unrotate(gtsam.Point3(0, 0, -9.81))
            if hasattr(g_body_vec, "x"):
                g_vec = np.array([g_body_vec.x(), g_body_vec.y(), g_body_vec.z()])
            else:
                g_vec = g_body_vec

            sf_true = data["accel_body"] - g_vec
            h_true_acc.append(sf_true)
            h_true_gyr.append(data["omega_body"])
            h_temp.append(data["temp"])

            if i > 600 and i % buffer_window == 0:
                mask = np.ones(21)

                # [수정] 데이터를 올바르게 패킹하여 전달 (튜플 리스트)
                packed_true = list(zip(h_true_acc, h_true_gyr))
                res = sysid.run(packed_true, h_meas, h_temp, acc_mask=mask, gyr_mask=mask)

                if res is not None:
                    # Bias 오차만으로 최적 파라미터 판단
                    acc_b_err = np.linalg.norm(res["acc_b"] - true_p_acc["bias"])
                    gyr_b_err = np.linalg.norm(res["gyr_b"] - true_p_gyr["bias"])
                    total_err = acc_b_err + gyr_b_err

                    if total_err < min_err:
                        min_err = total_err
                        best_p = copy.deepcopy(res)

                    if i % 600 == 0:
                        print(
                            f"  [Time {i * dt / 60:.1f}m] AccBiasErr: {acc_b_err:.4f} | GyrBiasErr: {gyr_b_err:.4f}"
                        )
                    curr_p = res
                else:
                    if i % 600 == 0:
                        print(f"  [Time {i * dt / 60:.1f}m] SysID Optimization Failed")

        # 4. Logging
        if i % buffer_window == 0:
            param_log["time"].append(i * dt)
            p = (
                curr_p
                if curr_p
                else {
                    "acc_b": np.zeros(3),
                    "acc_T_inv": np.eye(3),
                    "acc_k1": np.zeros(3),
                    "acc_k2": np.zeros(3),
                    "acc_h": np.zeros(3),
                    "gyr_b": np.zeros(3),
                    "gyr_T_inv": np.eye(3),
                    "gyr_k1": np.zeros(3),
                    "gyr_k2": np.zeros(3),
                    "gyr_h": np.zeros(3),
                }
            )
            # Acc
            param_log["acc"]["bias"].append(p["acc_b"])
            param_log["acc"]["scale"].append(np.diag(p["acc_T_inv"]))
            param_log["acc"]["temp_lin"].append(p["acc_k1"])
            param_log["acc"]["temp_non"].append(p["acc_k2"])
            param_log["acc"]["hyst"].append(p["acc_h"])
            # Gyr
            param_log["gyr"]["bias"].append(p["gyr_b"])
            param_log["gyr"]["scale"].append(np.diag(p["gyr_T_inv"]))
            param_log["gyr"]["temp_lin"].append(p["gyr_k1"])
            param_log["gyr"]["temp_non"].append(p["gyr_k2"])
            param_log["gyr"]["hyst"].append(p["gyr_h"])

        # 5. Correction
        g_body_vec = data["pose"].rotation().unrotate(gtsam.Point3(0, 0, -9.81))
        if hasattr(g_body_vec, "x"):
            g_vec = np.array([g_body_vec.x(), g_body_vec.y(), g_body_vec.z()])
        else:
            g_vec = g_body_vec

        if mode == "raw" or curr_p is None:
            corr_acc = meas_acc + g_vec
            corr_gyr = meas_gyr
        else:
            p = curr_p
            dt_t = data["temp"] - 20.0

            # --- Accel Correction ---
            sf_curr = data["accel_body"] - g_vec
            if i == 0:
                acc_diff = np.zeros(3)
            else:
                acc_diff = sf_curr - prev_sf
            prev_sf = sf_curr
            acc_h_sign = np.tanh(acc_diff * 10.0)

            acc_err = (
                p["acc_b"] + p["acc_k1"] * dt_t + p["acc_k2"] * (dt_t**2) + p["acc_h"] * acc_h_sign
            )
            corr_acc = p["acc_T_inv"] @ (meas_acc - acc_err)

            # --- Gyro Correction ---
            omega_curr = data["omega_body"]
            if i == 0:
                gyr_diff = np.zeros(3)
            else:
                gyr_diff = omega_curr - prev_omega
            prev_omega = omega_curr
            gyr_h_sign = np.tanh(gyr_diff * 10.0)

            gyr_err = (
                p["gyr_b"] + p["gyr_k1"] * dt_t + p["gyr_k2"] * (dt_t**2) + p["gyr_h"] * gyr_h_sign
            )
            corr_gyr = p["gyr_T_inv"] @ (meas_gyr - gyr_err)

        # 6. Integration
        pim.integrateMeasurement(corr_acc, corr_gyr, dt)
        nav = gtsam.NavState(curr_pose, curr_vel)
        next_s = pim.predict(nav, gtsam.imuBias.ConstantBias())
        curr_pose, curr_vel = next_s.pose(), next_s.velocity()
        pim.resetIntegration()
        poses.append(curr_pose)

    return poses, param_log, (best_p if mode == "online" else curr_p)


def run_improved_navigation():
    start_loc = (35.1796, 129.0756)
    dt = 0.1
    road_gen = RoadTrajectoryGenerator(location_point=start_loc, dist=8000)
    x, y, _ = road_gen.generate_path()
    sim_traj = apply_traffic_profile(road_gen.interpolate_trajectory(x, y, 20.0, dt), dt)

    # [설정] True Parameters with Scale Error
    true_acc = {
        "bias": np.array([0.1, -0.05, 0.02]),
        "scale": np.array([1.01, 0.99, 1.005]),
        "temp_lin": np.array([0.01, 0.01, 0.01]),
        "temp_non": np.array([0.0001, 0.0001, 0.0001]),
        "hyst": np.array([0.005, 0.003, 0.002]),
    }
    true_gyr = {
        "bias": np.array([0.01, -0.01, 0.005]),
        "scale": np.array([0.995, 1.005, 1.0]),
        "temp_lin": np.array([0.0001, 0.0001, 0.0001]),
        "temp_non": np.array([1e-5, 1e-5, 1e-5]),
        "hyst": np.array([0.0005, 0.0005, 0.0005]),
    }

    imu_cfg = {
        "accel_bias": true_acc["bias"],
        "accel_hysteresis": true_acc["hyst"],
        "accel_temp_coeff_linear": 0.01,
        "accel_temp_coeff_nonlinear": 0.0001,
        "accel_noise": 0.0001,
        "gyro_bias": true_gyr["bias"],
        "gyro_noise": 1e-5,
    }

    # Execute
    p_raw, _, _ = navigate(sim_traj, imu_cfg, "raw")
    p_on, log, b_p = navigate(sim_traj, imu_cfg, "online", true_p_acc=true_acc, true_p_gyr=true_gyr)
    p_val, _, _ = navigate(sim_traj, imu_cfg, "fixed", fixed_params=b_p)

    print(f"\n[Final Results - Accelerometer]")
    if b_p:
        print(f"  > Bias (Est): {b_p['acc_b']}")
        print(f"  > Bias (True): {true_acc['bias']}")
        print(f"  > Hyst (Est): {b_p['acc_h']}")
        print(f"  > Hyst (True): {true_acc['hyst']}")
    else:
        print("  > Calibration Failed")

    plot_all_calibration_results(log, true_acc, true_gyr, dt)

    # Position Error Plot
    def get_xyz(p):
        return np.array(
            [
                (pt.x(), pt.y()) if hasattr(pt, "x") else (pt[0], pt[1])
                for pt in [pp.translation() for pp in p]
            ]
        )

    gt_xy = get_xyz([d["pose"] for d in sim_traj])
    xy_raw, xy_on, xy_val = get_xyz(p_raw), get_xyz(p_on), get_xyz(p_val)
    min_l = min(len(gt_xy), len(xy_raw), len(xy_val))
    t_ax = np.arange(min_l) * dt / 60.0

    plt.figure(figsize=(10, 5))
    plt.plot(t_ax, np.linalg.norm(gt_xy[:min_l] - xy_raw[:min_l], axis=1), "r--", label="Raw")
    plt.plot(t_ax, np.linalg.norm(gt_xy[:min_l] - xy_on[:min_l], axis=1), "b-", label="Online")
    plt.plot(
        t_ax, np.linalg.norm(gt_xy[:min_l] - xy_val[:min_l], axis=1), "m-", label="Fixed", lw=2
    )
    plt.title("Position Error")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/pos_error.png")

    # Map
    m = folium.Map(location=start_loc, zoom_start=14)

    def add_trace(plist, color, name):
        m_deg = 111000.0
        lon_s = m_deg * np.cos(np.radians(start_loc[0]))
        coords = []
        for p in plist:
            t = p.translation()
            if hasattr(t, "x"):
                px, py = t.x(), t.y()
            else:
                px, py = t[0], t[1]
            coords.append([start_loc[0] + py / m_deg, start_loc[1] + px / lon_s])
        folium.PolyLine(coords[::10], color=color, weight=3, tooltip=name).add_to(m)

    add_trace([d["pose"] for d in sim_traj], "green", "GT")
    add_trace(p_raw, "red", "Raw")
    add_trace(p_val, "purple", "Proposed")
    m.save(f"{OUTPUT_DIR}/final_map_complete.html")
    print(f"Done. Results in '{OUTPUT_DIR}/'")

    plt.show()


if __name__ == "__main__":
    run_improved_navigation()
