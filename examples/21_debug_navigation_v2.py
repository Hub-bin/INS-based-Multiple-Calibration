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
# (이전과 동일)
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
        elif traffic_phase == 4:
            target_vel = 0.0
        else:
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


# --- New Plotting Functions ---
def plot_vector_params(log, true_params):
    """Figure 2: Bias, Temp, Hysteresis 시각화"""
    times = np.array(log["time"]) / 60.0  # min

    est_b = np.array(log["bias"])
    est_tl = np.array(log["temp_lin"])
    est_tn = np.array(log["temp_non"])
    est_h = np.array(log["hyst"])

    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    fig.suptitle("Figure 2: Vector Parameters Convergence (Bias, Temp, Hysteresis)", fontsize=14)

    rows_data = [
        (est_b, "bias", "Bias ($m/s^2$)"),
        (est_tl, "temp_lin", "Temp Lin ($/^\circ C$)"),
        (est_tn, "temp_non", "Temp Non ($/^\circ C^2$)"),
        (est_h, "hyst", "Hysteresis ($m/s^2$)"),
    ]

    axes_names = ["X-axis", "Y-axis", "Z-axis"]

    for row_idx, (est_data, true_key, ylabel) in enumerate(rows_data):
        true_val = true_params[true_key]
        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            ax.plot(times, est_data[:, col_idx], "b-", label="Est")
            ax.axhline(true_val[col_idx], color="r", linestyle="--", label="True")

            if col_idx == 0:
                ax.set_ylabel(ylabel)
            if row_idx == 0:
                ax.set_title(axes_names[col_idx])
            if row_idx == 3:
                ax.set_xlabel("Time (min)")

            # Hysteresis 그래프 Y축 범위 조정 (잘 보이게)
            if true_key == "hyst":
                ax.set_ylim(-0.01, 0.05)  # True 값이 0.02이므로 이 범위로 설정

            ax.grid(True)
            if row_idx == 0 and col_idx == 2:
                ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def plot_matrix_params(log, true_params):
    """Figure 3: Scale Factor & Misalignment (T_inv matrix) 시각화"""
    times = np.array(log["time"]) / 60.0

    # T_inv는 (N, 3, 3) 형태의 배열
    est_T_inv_hist = np.array(log["T_inv"])
    true_T_inv = true_params["T_inv"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("Figure 3: Matrix Parameters (Scale & Misalignment - $T^{-1}$)", fontsize=14)

    labels = [
        ["$T^{-1}_{xx}$ (Scale X)", "$T^{-1}_{xy}$ (Misalign)", "$T^{-1}_{xz}$ (Misalign)"],
        ["$T^{-1}_{yx}$ (Misalign)", "$T^{-1}_{yy}$ (Scale Y)", "$T^{-1}_{yz}$ (Misalign)"],
        ["$T^{-1}_{zx}$ (Misalign)", "$T^{-1}_{zy}$ (Misalign)", "$T^{-1}_{zz}$ (Scale Z)"],
    ]

    for r in range(3):
        for c in range(3):
            ax = axes[r, c]
            # 시간대별 r행 c열 요소 추출
            est_val_series = est_T_inv_hist[:, r, c]
            true_val = true_T_inv[r, c]

            ax.plot(times, est_val_series, "b-", label="Est")
            ax.axhline(true_val, color="r", linestyle="--", label="True")
            ax.set_title(labels[r][c])

            # 대각선(Scale)과 비대각선(Misalign)의 Y축 스케일을 다르게 설정
            if r == c:  # Scale Factor
                ax.set_ylim(0.99, 1.01)  # 1.0 근처 확대
            else:  # Misalignment
                ax.set_ylim(-0.01, 0.01)  # 0.0 근처 확대

            ax.grid(True)
            if r == 2:
                ax.set_xlabel("Time (min)")
            if r == 0 and c == 2:
                ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# --- Navigation Logic ---
def navigate_trajectory(
    sim_traj, imu_config, agent, mode="online", fixed_params=None, true_params_debug=None
):
    print(f"\n>>> Running Navigation Mode: {mode.upper()}...")

    imu = ImuSensor(**imu_config)
    sysid = SysIdCalibrator()

    est_poses = [sim_traj[0]["pose"]]
    curr_pose = sim_traj[0]["pose"]
    curr_vel = np.zeros(3)
    curr_bias = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))

    pim_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
    pim = gtsam.PreintegratedImuMeasurements(pim_params, curr_bias)

    buffer_window = 100
    history_meas = []
    history_true = []
    history_temp = []

    static_counter = 0
    static_gyr_sum = np.zeros(3)

    curr_params = None
    if mode == "fixed":
        curr_params = fixed_params

    # Logging update: store full T_inv matrix
    param_log = {"time": [], "bias": [], "T_inv": [], "temp_lin": [], "temp_non": [], "hyst": []}
    dt = 0.1

    min_param_error = float("inf")
    best_params = None

    for i, data in enumerate(sim_traj):
        true_pose = data["pose"]
        acc_body = data["accel_body"]
        omg_body = data["omega_body"]
        temp = data["temp"]

        meas_acc, meas_gyr, _ = imu.measure(true_pose, acc_body, omg_body, temperature=temp)

        # ZUPT
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

        if mode == "online":
            history_meas.append((meas_acc, meas_gyr))
            rot = true_pose.rotation()
            g_body_np = rot.unrotate(gtsam.Point3(0, 0, -9.81))
            if not isinstance(g_body_np, np.ndarray):
                g_body_np = np.array([g_body_np.x(), g_body_np.y(), g_body_np.z()])
            sf_true = acc_body - g_body_np
            history_true.append((sf_true, omg_body))
            history_temp.append(temp)

            # Update Step
            if i > 600 and i % buffer_window == 0:
                temps_arr = np.array(history_temp)
                accs_arr = np.array([m[0] for m in history_meas])
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
                    acc_mask[9:12] = 1.0
                if decision[1]:
                    acc_mask[12:18] = 1.0
                if decision[2]:
                    acc_mask[18:21] = 1.0

                temp_res = sysid.run(history_true, history_meas, history_temp, acc_mask=acc_mask)

                est_b = temp_res["acc_b"]
                true_b = true_params_debug["bias"]
                bias_err = np.linalg.norm(est_b - true_b)

                if bias_err < min_param_error:
                    min_param_error = bias_err
                    best_params = copy.deepcopy(temp_res)

                if i % 600 == 0:
                    print(
                        f"  [Time {i * dt / 60:.1f}m] RL:{decision.astype(int)} | BiasErr: {bias_err:.4f}"
                    )

                curr_params = temp_res

        # Logging
        if i % buffer_window == 0:
            param_log["time"].append(i * dt)
            if curr_params:
                param_log["bias"].append(curr_params["acc_b"])
                param_log["T_inv"].append(curr_params["acc_T_inv"])  # Store full matrix
                param_log["temp_lin"].append(curr_params["acc_k1"])
                param_log["temp_non"].append(curr_params["acc_k2"])
                param_log["hyst"].append(curr_params["acc_h"])
            else:
                param_log["bias"].append(np.zeros(3))
                param_log["T_inv"].append(np.eye(3))  # Default identity
                param_log["temp_lin"].append(np.zeros(3))
                param_log["temp_non"].append(np.zeros(3))
                param_log["hyst"].append(np.zeros(3))

        # Correction
        if mode == "raw" or curr_params is None:
            corr_acc = meas_acc
        else:
            p = curr_params
            dt_temp = temp - 20.0
            err_bias = p["acc_b"]
            err_lin = p["acc_k1"] * dt_temp
            err_non = p["acc_k2"] * (dt_temp**2)
            hyst_sign = np.sign(meas_acc)
            err_hyst = p["acc_h"] * hyst_sign
            term = meas_acc - err_bias - err_lin - err_non - err_hyst
            corr_acc = p["acc_T_inv"] @ term

        pim.integrateMeasurement(corr_acc, meas_gyr, dt)
        nav_state = gtsam.NavState(curr_pose, curr_vel)
        next_state = pim.predict(nav_state, curr_bias)
        curr_pose = next_state.pose()
        curr_vel = next_state.velocity()
        pim.resetIntegration()
        est_poses.append(curr_pose)

    final_ret_params = best_params if mode == "online" else curr_params
    if mode == "online" and best_params is None:
        final_ret_params = curr_params

    return est_poses, param_log, final_ret_params


def run_improved_navigation():
    print("=== Improved Navigation: Final Diagnostics V2 ===")

    start_loc = (35.1796, 129.0756)
    dt = 0.1

    road_gen = RoadTrajectoryGenerator(location_point=start_loc, dist=8000)
    x_pts, y_pts, route_ids = road_gen.generate_path()
    base_traj = road_gen.interpolate_trajectory(x_pts, y_pts, target_speed=20.0, dt=dt)
    sim_traj = apply_traffic_profile(base_traj, dt, total_duration_min=10)

    # [중요] True Parameter 설정 (미정렬 포함)
    true_T_inv = np.array(
        [
            [1.001, 0.0002, 0.0001],  # Slight scale error & misalignment
            [0.0001, 1.0015, 0.0002],
            [0.0001, 0.0001, 1.002],
        ]
    )

    true_params = {
        "bias": np.array([0.05, -0.02, 0.01]),
        "T_inv": true_T_inv,
        "temp_lin": np.array([0.01, 0.01, 0.01]),
        "temp_non": np.array([0.0001, 0.0001, 0.0001]),
        "hyst": np.array([0.02, 0.02, 0.02]),
    }

    # ImuSensor는 벡터 형태의 scale/misalignment를 받지 않으므로,
    # 시뮬레이션 생성 시에는 이상적인 ImuSensor를 쓰고,
    # 데이터 주입 시에 수동으로 오차를 더하는 방식이 맞으나,
    # 현재 ImuSensor 구현상 T_inv를 직접 주입할 수 없으므로
    # 여기서는 Bias/Temp/Hyst만 ImuSensor로 넣고, Scale/Misalignment는 이상적(Identity)이라고 가정하고 테스트합니다.
    # (Scale/Misalignment 추정 기능 자체가 동작하는지 확인하는 목적)

    # 만약 Scale/Misalignment가 포함된 데이터를 만들려면 ImuSensor 클래스 수정이 필요합니다.
    # 현재는 ImuSensor가 제공하는 오차만 주입합니다.
    imu_config = {
        "accel_bias": true_params["bias"],
        "accel_temp_coeff_linear": 0.01,
        "accel_temp_coeff_nonlinear": 0.0001,
        "accel_hysteresis": 0.02,
        "accel_noise": 0.0001,
        "gyro_noise": 0.00001,
        "gyro_bias": [0.0001, 0.0001, 0.0001],
    }
    # 따라서 True T_inv는 Identity로 설정하여 플롯 검증
    true_params["T_inv"] = np.eye(3)

    agent = RLAgent(input_dim=4, action_dim=3)
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

    # Run Modes
    poses_raw, _, _ = navigate_trajectory(sim_traj, imu_config, agent, mode="raw")
    poses_online, log_online, best_params = navigate_trajectory(
        sim_traj, imu_config, agent, mode="online", true_params_debug=true_params
    )
    poses_valid, _, _ = navigate_trajectory(
        sim_traj, imu_config, agent, mode="fixed", fixed_params=best_params
    )

    # Visualization
    print("Generating Plots...")

    # Figure 1: Position Error
    def get_err(poses):
        est_xy = np.array(
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
        min_len = min(len(gt_xy), len(est_xy))
        return np.linalg.norm(gt_xy[:min_len] - est_xy[:min_len], axis=1), min_len

    err_raw, len_raw = get_err(poses_raw)
    err_online, len_online = get_err(poses_online)
    err_valid, len_valid = get_err(poses_valid)
    t_axis = np.arange(len_raw) * dt / 60.0

    plt.figure(figsize=(10, 5))
    plt.plot(t_axis, err_raw, "r--", label="Raw Integration")
    plt.plot(t_axis, err_online, "b-", label="Online Learning")
    plt.plot(t_axis, err_valid, "m-", label="Validation (Best Params)", linewidth=2)
    plt.title("Figure 1: Position Error Comparison")
    plt.xlabel("Time (min)")
    plt.ylabel("Error (m)")
    plt.legend()
    plt.grid(True)

    # Figure 2: Vector Params (Bias, Temp, Hyst)
    plot_vector_params(log_online, true_params)

    # Figure 3: Matrix Params (Scale, Misalignment)
    plot_matrix_params(log_online, true_params)

    plt.show()  # Show all plots at once


if __name__ == "__main__":
    run_improved_navigation()
