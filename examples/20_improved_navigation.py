import sys
import os

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


# --- Helper: 교통 상황 생성 ---
def apply_traffic_profile(trajectory, dt, total_duration_min=10):
    total_steps = int(total_duration_min * 60 / dt)
    n_points = len(trajectory)
    new_traj = []
    curr_idx = 0
    curr_vel = 0.0

    print(f"Generating Traffic Profile for {total_duration_min} mins ({total_steps} steps)...")
    for i in range(total_steps):
        t = i * dt
        traffic_phase = int(t / 60.0) % 5
        if traffic_phase == 0 or traffic_phase == 2:  # Stop & Go
            target_vel = 5.0 + 5.0 * np.sin(t * 0.5)
            if target_vel < 0:
                target_vel = 0
        elif traffic_phase == 4:  # Stop
            target_vel = 0.0
        else:  # Cruising
            target_vel = 15.0 + np.random.normal(0, 0.5)

        acc = target_vel - curr_vel
        acc = np.clip(acc, -3.0, 2.0)
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
        pose_interp = gtsam.Pose3(r_interp, t_interp)

        acc_body = np.array([acc, 0.0, 0.0])
        orig_omega = trajectory[idx_int]["omega_body"]
        scaled_omega = orig_omega * (curr_vel / 20.0)

        new_traj.append(
            {
                "pose": pose_interp,
                "accel_body": acc_body,
                "omega_body": scaled_omega,
                "vel_world": curr_vel,
                "temp": 20.0 + (40.0 * (i / total_steps)),
            }
        )
    return new_traj


# --- Core Logic: Navigation Simulation Loop ---
def navigate_trajectory(sim_traj, imu_config, agent, mode="online", fixed_params=None, dt=0.1):
    """
    mode: 'online' (RL+SysID 수행), 'fixed' (고정 파라미터 적용), 'raw' (보정 없음)
    """
    print(f"\n>>> Running Navigation Mode: {mode.upper()}...")

    # IMU Reset (Stateful Hysteresis 초기화)
    imu = ImuSensor(**imu_config)
    sysid = SysIdCalibrator()

    # Navigation States
    est_poses = [sim_traj[0]["pose"]]
    curr_pose = sim_traj[0]["pose"]
    curr_vel = np.zeros(3)
    curr_bias = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))

    pim_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
    pim = gtsam.PreintegratedImuMeasurements(pim_params, curr_bias)

    # Buffers
    buffer_window = 100
    history_meas = []
    history_true = []
    history_temp = []

    # ZUPT vars
    static_counter = 0
    static_gyr_sum = np.zeros(3)

    # Parameter Tracking
    curr_params = None
    if mode == "fixed" and fixed_params is not None:
        curr_params = fixed_params

    param_log = {"time": [], "bias": [], "scale": [], "temp_lin": [], "temp_non": [], "hyst": []}

    total_steps = len(sim_traj)

    for i, data in enumerate(sim_traj):
        true_pose = data["pose"]
        acc_body = data["accel_body"]
        omg_body = data["omega_body"]
        temp = data["temp"]

        meas_acc, meas_gyr, _ = imu.measure(true_pose, acc_body, omg_body, temperature=temp)

        # --- ZUPT Logic (Common for all modes except maybe pure raw, but let's apply to all for fair comparison of sensor calibration) ---
        # Raw 모드는 '순수 항법'이므로 ZUPT도 끄는 것이 맞지만,
        # 여기서는 "센서 캘리브레이션 유무"의 차이를 보기 위해 ZUPT는 기본 항법 로직으로 모두 적용
        is_stopped = data["vel_world"] < 0.05
        if is_stopped:
            static_counter += 1
            static_gyr_sum += meas_gyr
            curr_vel = np.zeros(3)
            if static_counter > 10:
                avg_gyr = static_gyr_sum / static_counter
                curr_bias = gtsam.imuBias.ConstantBias(np.zeros(3), avg_gyr)
                pim.resetIntegrationAndSetBias(curr_bias)
        else:
            static_counter = 0
            static_gyr_sum = np.zeros(3)

        # History Accumulation
        if mode == "online":
            history_meas.append((meas_acc, meas_gyr))
            rot = true_pose.rotation()
            g_body_np = rot.unrotate(gtsam.Point3(0, 0, -9.81))
            if not isinstance(g_body_np, np.ndarray):
                g_body_np = np.array([g_body_np.x(), g_body_np.y(), g_body_np.z()])
            sf_true = acc_body - g_body_np
            history_true.append((sf_true, omg_body))
            history_temp.append(temp)

            # --- Online Calibration Update ---
            if i > 600 and i % buffer_window == 0:
                # RL Action
                temps_arr = np.array(history_temp)
                accs_arr = np.array([m[0] for m in history_meas])
                # Simple Normalization
                state = [
                    (np.mean(temps_arr) - 20) / 40.0,
                    np.std(temps_arr) * 10.0,
                    np.std(accs_arr),
                    np.mean(np.abs(np.diff(accs_arr, axis=0))) * 10.0,
                ]

                action, _ = agent.get_action(state, deterministic=True)
                decision = action > 0.0

                acc_mask = np.zeros(21)
                if decision[0]:
                    acc_mask[9:12] = 1.0  # Bias
                if decision[1]:
                    acc_mask[12:18] = 1.0  # Temp (Lin+Non)
                if decision[2]:
                    acc_mask[18:21] = 1.0  # Hyst

                # SysID (Use Accumulated History)
                temp_res = sysid.run(history_true, history_meas, history_temp, acc_mask=acc_mask)

                # Safe Guard
                if np.linalg.norm(temp_res["acc_b"]) < 0.5:
                    curr_params = temp_res

                # Do NOT clear history for iterative improvement

        # --- Parameter Logging ---
        if i % buffer_window == 0:
            param_log["time"].append(i * dt)
            if curr_params:
                param_log["bias"].append(curr_params["acc_b"])
                param_log["scale"].append(np.diag(curr_params["acc_T_inv"]))
                param_log["temp_lin"].append(curr_params["acc_k1"])
                param_log["temp_non"].append(curr_params["acc_k2"])
                param_log["hyst"].append(curr_params["acc_h"])
            else:
                param_log["bias"].append(np.zeros(3))
                param_log["scale"].append(np.ones(3))
                param_log["temp_lin"].append(np.zeros(3))
                param_log["temp_non"].append(np.zeros(3))
                param_log["hyst"].append(np.zeros(3))

        # --- Correction Application ---
        if mode == "raw" or curr_params is None:
            corr_acc = meas_acc
        else:
            p = curr_params
            dt_temp = temp - 20.0

            err_bias = p["acc_b"]
            err_lin = p["acc_k1"] * dt_temp
            err_non = p["acc_k2"] * (dt_temp**2)
            hyst_sign = np.sign(meas_acc)  # Simple sign
            err_hyst = p["acc_h"] * hyst_sign

            # Calibration Model: True = T_inv * (Meas - Bias - Temp - Hyst)
            term = meas_acc - err_bias - err_lin - err_non - err_hyst
            corr_acc = p["acc_T_inv"] @ term

        pim.integrateMeasurement(corr_acc, meas_gyr, dt)
        nav_state = gtsam.NavState(curr_pose, curr_vel)
        next_state = pim.predict(nav_state, curr_bias)
        curr_pose = next_state.pose()
        curr_vel = next_state.velocity()
        pim.resetIntegration()
        est_poses.append(curr_pose)

        if i % 2000 == 0:
            print(f"  Step {i}/{total_steps} done.")

    return est_poses, param_log, curr_params


def run_improved_navigation():
    print("=== Improved Navigation: All-in-One Analysis ===")

    start_loc = (35.1796, 129.0756)
    dt = 0.1

    # 1. Trajectory Generation
    road_gen = RoadTrajectoryGenerator(location_point=start_loc, dist=8000)
    x_pts, y_pts, route_ids = road_gen.generate_path()
    base_traj = road_gen.interpolate_trajectory(x_pts, y_pts, target_speed=20.0, dt=dt)
    sim_traj = apply_traffic_profile(base_traj, dt, total_duration_min=10)

    # 2. Sensor Config (1-mil grade)
    true_params = {
        "bias": np.array([0.05, -0.02, 0.01]),
        "temp_lin": np.array([0.01, 0.01, 0.01]),
        "temp_non": np.array([0.0001, 0.0001, 0.0001]),
        "hyst": np.array([0.02, 0.02, 0.02]),
    }

    imu_config = {
        "accel_bias": true_params["bias"],
        "accel_temp_coeff_linear": 0.01,
        "accel_temp_coeff_nonlinear": 0.0001,
        "accel_hysteresis": 0.02,
        "accel_noise": 0.0001,
        "gyro_noise": 0.00001,
        "gyro_bias": [0.0001, 0.0001, 0.0001],
    }

    # 3. Agent & SysID Setup
    agent = RLAgent(input_dim=4, action_dim=3)
    # Load Pre-trained Weights (Simulation)
    with torch.no_grad():
        agent.policy.fc1.weight.fill_(0.0)
        agent.policy.fc1.bias.fill_(0.0)
        agent.policy.fc1.weight[0, 1] = 5.0
        agent.policy.fc1.weight[1, 3] = 5.0
        agent.policy.fc2.weight.fill_(0.0)
        agent.policy.fc2.bias.fill_(0.0)
        agent.policy.fc2.weight[0, 0] = 1.0
        agent.policy.fc2.weight[1, 1] = 1.0
        agent.policy.mu_head.weight.fill_(0.0)
        agent.policy.mu_head.bias.data = torch.tensor([2.0, -2.0, -2.0])
        agent.policy.mu_head.weight[1, 0] = 5.0
        agent.policy.mu_head.weight[2, 1] = 5.0

    # 4. Run Scenarios
    # A. Raw (No Calibration)
    poses_raw, _, _ = navigate_trajectory(sim_traj, imu_config, agent, mode="raw")

    # B. Online (RL + SysID Learning)
    poses_online, log_online, final_params = navigate_trajectory(
        sim_traj, imu_config, agent, mode="online"
    )

    # C. Validation (Apply Final Params from Start)
    poses_valid, _, _ = navigate_trajectory(
        sim_traj, imu_config, agent, mode="fixed", fixed_params=final_params
    )

    # 5. Visualization
    # Map
    print("Generating Map...")
    m = folium.Map(location=start_loc, zoom_start=14)

    def get_latlon(poses):
        m_per_deg_lat = 111000.0
        m_per_deg_lon = 111000.0 * np.cos(np.radians(start_loc[0]))
        path = []
        for p in poses:
            pos = p.translation()
            if isinstance(pos, np.ndarray):
                px, py = pos[0], pos[1]
            else:
                px, py = pos.x(), pos.y()
            path.append([start_loc[0] + py / m_per_deg_lat, start_loc[1] + px / m_per_deg_lon])
        return path

    gt_path = []  # Extract GT from traj
    for p in sim_traj:
        pos = p["pose"].translation()
        if isinstance(pos, np.ndarray):
            px, py = pos[0], pos[1]
        else:
            px, py = pos.x(), pos.y()
        m_per_deg_lat = 111000.0
        m_per_deg_lon = 111000.0 * np.cos(np.radians(start_loc[0]))
        gt_path.append([start_loc[0] + py / m_per_deg_lat, start_loc[1] + px / m_per_deg_lon])

    path_raw = get_latlon(poses_raw)
    path_online = get_latlon(poses_online)
    path_valid = get_latlon(poses_valid)

    folium.PolyLine(gt_path, color="green", weight=5, opacity=0.6, tooltip="Ground Truth").add_to(m)
    folium.PolyLine(
        path_raw, color="red", weight=2, opacity=0.6, dash_array="5, 10", tooltip="Raw Integration"
    ).add_to(m)
    folium.PolyLine(
        path_online, color="blue", weight=3, opacity=0.8, tooltip="Online Learning"
    ).add_to(m)
    folium.PolyLine(
        path_valid, color="purple", weight=3, opacity=0.8, tooltip="Validation (Final Params)"
    ).add_to(m)

    folium.Marker(gt_path[0], popup="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(gt_path[-1], popup="End", icon=folium.Icon(color="red")).add_to(m)
    m.save("navigation_all_in_one.html")
    print("Map saved.")

    # Figures
    # 5.1 Position Error
    def get_xy(poses):
        return np.array(
            [
                (p.translation()[0], p.translation()[1])
                if isinstance(p.translation(), np.ndarray)
                else (p.x(), p.y())
                for p in poses
            ]
        )

    gt_xy = np.array(
        [
            (p["pose"].translation()[0], p["pose"].translation()[1])
            if isinstance(p["pose"].translation(), np.ndarray)
            else (p["pose"].x(), p["pose"].y())
            for p in sim_traj
        ]
    )

    xy_raw = get_xy(poses_raw)
    xy_online = get_xy(poses_online)
    xy_valid = get_xy(poses_valid)
    min_len = min(len(gt_xy), len(xy_raw))

    t_axis = np.arange(min_len) * dt / 60.0

    err_raw = np.linalg.norm(gt_xy[:min_len] - xy_raw[:min_len], axis=1)
    err_online = np.linalg.norm(gt_xy[:min_len] - xy_online[:min_len], axis=1)
    err_valid = np.linalg.norm(gt_xy[:min_len] - xy_valid[:min_len], axis=1)

    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(t_axis, err_raw, "r--", label="Raw")
    plt.plot(t_axis, err_online, "b-", label="Online Learning")
    plt.plot(t_axis, err_valid, "m-", label="Validation (Final Params)")
    plt.title("Position Error Comparison")
    plt.xlabel("Time (min)")
    plt.ylabel("Error (m)")
    plt.legend()
    plt.grid(True)

    # 5.2 Calibration Parameters
    times = np.array(log_online["time"]) / 60.0

    fig2, axes = plt.subplots(5, 1, figsize=(10, 15))

    # Bias
    est_b = np.array(log_online["bias"])
    axes[0].plot(times, est_b[:, 0], "r", label="Est X")
    axes[0].plot(times, est_b[:, 1], "g", label="Est Y")
    axes[0].plot(times, est_b[:, 2], "b", label="Est Z")
    axes[0].axhline(true_params["bias"][0], color="r", ls="--")
    axes[0].axhline(true_params["bias"][1], color="g", ls="--")
    axes[0].axhline(true_params["bias"][2], color="b", ls="--")
    axes[0].set_title("1. Bias Estimation")
    axes[0].grid(True)

    # Scale
    est_s = np.array(log_online["scale"])
    axes[1].plot(times, est_s[:, 0], "r")
    axes[1].plot(times, est_s[:, 1], "g")
    axes[1].plot(times, est_s[:, 2], "b")
    axes[1].axhline(1.0, color="k", ls="--")
    axes[1].set_title("2. Scale Factor (T_inv diag)")
    axes[1].grid(True)

    # Temp Lin
    est_tl = np.array(log_online["temp_lin"])
    axes[2].plot(times, est_tl[:, 0], "r")
    axes[2].plot(times, est_tl[:, 1], "g")
    axes[2].plot(times, est_tl[:, 2], "b")
    axes[2].axhline(true_params["temp_lin"][0], color="r", ls="--")
    axes[2].set_title("3. Temp Linear Coeff")
    axes[2].grid(True)

    # Temp Non
    est_tn = np.array(log_online["temp_non"])
    axes[3].plot(times, est_tn[:, 0], "r")
    axes[3].plot(times, est_tn[:, 1], "g")
    axes[3].plot(times, est_tn[:, 2], "b")
    axes[3].axhline(true_params["temp_non"][0], color="r", ls="--")
    axes[3].set_title("4. Temp Nonlinear Coeff")
    axes[3].grid(True)

    # Hysteresis
    est_h = np.array(log_online["hyst"])
    axes[4].plot(times, est_h[:, 0], "r")
    axes[4].plot(times, est_h[:, 1], "g")
    axes[4].plot(times, est_h[:, 2], "b")
    axes[4].axhline(true_params["hyst"][0], color="r", ls="--")
    axes[4].set_title("5. Hysteresis Factor")
    axes[4].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_improved_navigation()
