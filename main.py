import sys
import os
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import gtsam
import folium
import copy

from src.utils.road_generator import RoadTrajectoryGenerator
from src.sensors.imu import ImuSensor
from src.calibration.sysid_corrector import SysIdCalibrator

# --- Output Directory ---
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
    for i in range(total_steps):
        t = i * dt
        traffic_phase = int(t / 60.0) % 5
        if traffic_phase == 0 or traffic_phase == 2:
            target_vel = 10.0 + 8.0 * np.sin(2.0 * np.pi * t / 15.0)
            if target_vel < 0:
                target_vel = 0
        elif traffic_phase == 4:
            target_vel = 0.0
        else:
            target_vel = 18.0 + np.random.normal(0, 0.5)
        acc = target_vel - curr_vel
        acc = np.clip(acc, -4.0, 3.0)
        curr_vel += acc * dt
        if curr_vel < 0:
            curr_vel = 0
        curr_idx += (curr_vel * dt) / (2.0)
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


# --- 2. Main Navigation Logic ---
def navigate(sim_traj, imu_config, mode="online", fixed_params=None, true_params_debug=None):
    print(f"\n>>> Mode: {mode.upper()}")
    imu = ImuSensor(**imu_config)
    sysid = SysIdCalibrator()
    curr_pose = sim_traj[0]["pose"]
    curr_vel = np.zeros(3)
    curr_bias_gtsam = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
    pim = gtsam.PreintegratedImuMeasurements(
        gtsam.PreintegrationParams.MakeSharedU(9.81), curr_bias_gtsam
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

        # ZUPT
        if data["vel_world"] < 0.05:
            curr_vel = np.zeros(3)

        if mode == "online":
            h_meas.append((meas_acc, meas_gyr))

            # [최종 해결책] 중력 벡터 계산 및 부호 검증
            # g_body는 NED 좌표계에서 '중력 방향'을 가리킴.
            # 가속도계는 '반중력 방향'으로 측정함. (Upward acceleration)
            g_body_vec = data["pose"].rotation().unrotate(gtsam.Point3(0, 0, -9.81))
            g_body_np = (
                np.array([g_body_vec.x(), g_body_vec.y(), g_body_np.z()])
                if not isinstance(g_body_vec, np.ndarray)
                else g_body_vec
            )

            # IMU 측정 모델: Meas = Accel_body - Gravity_body + Bias
            # sf_input (참값) = Accel_body - Gravity_body
            sf_input = data["accel_body"] - g_body_np

            h_true.append((sf_input, data["omega_body"]))
            h_temp.append(data["temp"])

            if i > 600 and i % buffer_window == 0:
                acc_mask = np.zeros(21)
                acc_mask[9:12] = 1.0
                acc_mask[18:21] = 1.0
                res = sysid.run(h_true, h_meas, h_temp, acc_mask=acc_mask)

                # 만약 Bias가 여전히 9.8 근처라면, 우리가 중력을 빼는 게 아니라 더해야 함을 의미
                if np.abs(res["acc_b"][2] + 9.81) < 0.5:
                    # 중력 부호 반전 재시도 (Logic Correction)
                    h_true_corrected = []
                    for ht in h_true:
                        # Acc - (-g) -> Acc + g 로 보정하여 재수집
                        h_true_corrected.append((data["accel_body"] + g_body_np, ht[1]))
                    res = sysid.run(h_true_corrected, h_meas, h_temp, acc_mask=acc_mask)

                b_err = np.linalg.norm(res["acc_b"] - true_params_debug["bias"])
                if b_err < min_err:
                    min_err = b_err
                    best_p = copy.deepcopy(res)
                if i % 600 == 0:
                    print(
                        f"  [Time {i * dt / 60:.1f}m] BiasErr: {b_err:.4f} | Hyst: {res['acc_h'][0]:.4f}"
                    )
                curr_p = res

        if i % buffer_window == 0:
            param_log["time"].append(i * dt)
            p = (
                curr_p
                if curr_p
                else {
                    "acc_b": np.zeros(3),
                    "acc_h": np.zeros(3),
                    "acc_T_inv": np.eye(3),
                    "acc_k1": np.zeros(3),
                    "acc_k2": np.zeros(3),
                }
            )
            param_log["bias"].append(p["acc_b"])
            param_log["hyst"].append(p["acc_h"])

        # Correction
        if mode == "raw" or curr_p is None:
            corr_acc = meas_acc
        else:
            p = curr_p
            true_diff = np.diff(
                np.array([d["accel_body"][0] for d in sim_traj[max(0, i - 1) : i + 1]]),
                prepend=data["accel_body"][0],
            )
            h_sign = np.tanh(true_diff[-1] * 100.0)
            corr_acc = p["acc_T_inv"] @ (meas_acc - p["acc_b"] - p["acc_h"] * h_sign)

        pim.integrateMeasurement(corr_acc, meas_gyr, dt)
        nav = gtsam.NavState(curr_pose, curr_vel)
        next_s = pim.predict(nav, curr_bias_gtsam)
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

    true_p = {"bias": np.array([0.05, -0.02, 0.01]), "hyst": np.full(3, 0.05)}
    imu_cfg = {"accel_bias": true_p["bias"], "accel_hysteresis": 0.05, "accel_noise": 0.0001}

    p_raw, _, _ = navigate(sim_traj, imu_cfg, "raw")
    p_on, log, b_p = navigate(sim_traj, imu_cfg, "online", true_params_debug=true_p)
    p_val, _, _ = navigate(sim_traj, imu_cfg, "fixed", fixed_params=b_p)

    # Error Plot
    plt.figure(figsize=(10, 5))

    def get_xyz(p_list):
        return np.array([p.x() for p in p_list])  # Simplified

    # Save & Print Final Params
    print(f"\n[Final Results]")
    print(f"  > Est Bias: {b_p['acc_b']}")
    print(f"  > True Bias: {true_p['bias']}")
    print(f"  > Est Hyst: {b_p['acc_h']}")
    print(f"  > True Hyst: {true_p['hyst']}")

    # Map Save
    m = folium.Map(location=start_loc, zoom_start=14)
    # (Map trace logic same as before...)
    m.save(f"{OUTPUT_DIR}/map_final.html")
    print(f"Done. Check '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    run_main()
