import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import gtsam
import folium
import torch

from src.utils.road_generator import RoadTrajectoryGenerator
from src.dynamics.ground import GroundVehicle
from src.sensors.imu import ImuSensor
from src.calibration.sysid_corrector import SysIdCalibrator
from src.calibration.rl_agent import RLAgent


# --- Helper: 현실적인 속도 프로파일 생성 (Traffic Simulation) ---
def apply_traffic_profile(trajectory, dt, total_duration_min=40):
    """
    경로에 가감속(Stop & Go)과 정속 주행을 섞어서 40분짜리 데이터를 만듦
    """
    total_steps = int(total_duration_min * 60 / dt)
    n_points = len(trajectory)

    new_traj = []

    curr_idx = 0
    curr_vel = 0.0

    print(f"Generating Traffic Profile for {total_duration_min} mins ({total_steps} steps)...")

    for i in range(total_steps):
        t = i * dt

        # 1분(600스텝)마다 교통 상황 변경
        traffic_phase = int(t / 60.0) % 5

        if traffic_phase == 0 or traffic_phase == 2:
            # 정체 구간 (Stop & Go)
            target_vel = 5.0 + 5.0 * np.sin(t * 0.5)
            if target_vel < 0:
                target_vel = 0
        elif traffic_phase == 4:
            # 신호 대기 (완전 정지)
            target_vel = 0.0
        else:
            # 원활한 구간 (Cruising)
            target_vel = 15.0 + np.random.normal(0, 0.5)

        acc = target_vel - curr_vel
        acc = np.clip(acc, -3.0, 2.0)

        curr_vel += acc * dt
        if curr_vel < 0:
            curr_vel = 0

        step_dist = curr_vel * dt
        orig_step_dist = 20.0 * 0.1

        idx_increment = step_dist / orig_step_dist
        curr_idx += idx_increment

        if curr_idx >= n_points - 1:
            curr_idx = n_points - 1
            curr_vel = 0.0

        idx_int = int(curr_idx)
        alpha = curr_idx - idx_int

        p1 = trajectory[idx_int]["pose"]
        p2 = trajectory[min(idx_int + 1, n_points - 1)]["pose"]

        # Translation interpolation
        t1 = p1.translation()
        t2 = p2.translation()

        # t1, t2가 numpy array일 수도 있고 Point3일 수도 있음
        if not isinstance(t1, np.ndarray):
            t1 = np.array([t1.x(), t1.y(), t1.z()])
        if not isinstance(t2, np.ndarray):
            t2 = np.array([t2.x(), t2.y(), t2.z()])

        t_interp = t1 * (1 - alpha) + t2 * alpha

        r1 = p1.rotation()
        r2 = p2.rotation()
        r_interp = r1.slerp(alpha, r2)

        # t_interp는 numpy array임. Pose3 생성자에 그대로 넘김.
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


def run_navigation_simulation():
    print("=== Long-term Navigation Test (Busan City Hall -> Haeundae) ===")

    # 1. 지도 경로 생성
    start_loc = (35.1796, 129.0756)
    dt = 0.1

    road_gen = RoadTrajectoryGenerator(location_point=start_loc, dist=8000)
    print("Generating Route on Map...")
    x_pts, y_pts, route_ids = road_gen.generate_path()

    base_traj = road_gen.interpolate_trajectory(x_pts, y_pts, target_speed=20.0, dt=dt)

    # 40분 주행 데이터 생성
    sim_traj = apply_traffic_profile(base_traj, dt, total_duration_min=40)

    # 2. IMU 센서 설정
    accel_noise = 0.002
    imu = ImuSensor(
        accel_bias=[0.3, -0.1, 0.1],
        accel_temp_coeff_linear=0.05,
        accel_temp_coeff_nonlinear=0.001,
        accel_hysteresis=0.15,
        accel_noise=accel_noise,
        gyro_noise=0.0001,
    )

    sysid = SysIdCalibrator()

    # 3. RL Agent 모사 (가중치 강제 주입)
    # 입력: [Temp_Norm_Mean, Temp_Norm_Std, Acc_Std, Acc_Jerk]
    # 출력: [Bias, Temp, Hysteresis]
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

    print("RL Agent Loaded (Simulated Trained Weights).")

    # 4. 주행 시뮬레이션
    est_poses = [sim_traj[0]["pose"]]
    curr_pose = sim_traj[0]["pose"]
    curr_vel = gtsam.Point3(0, 0, 0)
    curr_bias = gtsam.imuBias.ConstantBias()

    pim_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
    pim = gtsam.PreintegratedImuMeasurements(pim_params, curr_bias)

    buffer_window = 100
    history_meas = []
    history_true = []
    history_temp = []

    # 보정 파라미터 초기화
    curr_params = None

    print("Starting Drive...")
    for i, data in enumerate(sim_traj):
        if i % 6000 == 0:
            print(f"  Time: {i * dt / 60:.1f} min / 40.0 min")

        true_pose = data["pose"]
        acc_body = data["accel_body"]
        omg_body = data["omega_body"]
        temp = data["temp"]

        meas_acc, meas_gyr, _ = imu.measure(true_pose, acc_body, omg_body, temperature=temp)

        history_meas.append((meas_acc, meas_gyr))

        rot = true_pose.rotation()
        g_body_np = rot.unrotate(gtsam.Point3(0, 0, -9.81))

        # g_body_np 타입 체크
        if not isinstance(g_body_np, np.ndarray):
            g_body_np = np.array([g_body_np.x(), g_body_np.y(), g_body_np.z()])

        sf_true = acc_body - g_body_np
        history_true.append((sf_true, omg_body))
        history_temp.append(temp)

        # 주기적 교정 (Every 10 seconds)
        if i > 0 and i % buffer_window == 0:
            temps_arr = np.array(history_temp)
            accs_arr = np.array([m[0] for m in history_meas])

            norm_temp_mean = (np.mean(temps_arr) - 20) / 40.0
            norm_temp_std = np.std(temps_arr) * 10.0
            norm_acc_std = np.std(accs_arr)
            norm_acc_jerk = np.mean(np.abs(np.diff(accs_arr, axis=0))) * 10.0

            state = [norm_temp_mean, norm_temp_std, norm_acc_std, norm_acc_jerk]

            # [수정] action, log_prob 2개 반환
            action, _ = agent.get_action(state, deterministic=True)
            decision = action > 0.0

            acc_mask = np.zeros(21)
            if decision[0]:
                acc_mask[9:12] = 1.0  # Bias
            if decision[1]:
                acc_mask[12:18] = 1.0  # Temp
            if decision[2]:
                acc_mask[18:21] = 1.0  # Hysteresis

            # SysID 실행
            calib_res = sysid.run(history_true, history_meas, history_temp, acc_mask=acc_mask)
            curr_params = calib_res

            history_meas = []
            history_true = []
            history_temp = []

        # --- Dead Reckoning ---
        if curr_params is None:
            corr_acc = meas_acc
            corr_gyr = meas_gyr
        else:
            p = curr_params
            dt_temp = temp - 20.0

            err_bias = p["acc_b"]
            err_temp = (p["acc_k1"] * dt_temp) + (p["acc_k2"] * (dt_temp**2))

            hyst_sign = np.sign(meas_acc)
            err_hyst = p["acc_h"] * hyst_sign

            corr_acc = p["acc_T_inv"] @ (meas_acc - err_bias - err_temp - err_hyst)
            corr_gyr = meas_gyr

        pim.integrateMeasurement(corr_acc, corr_gyr, dt)

        nav_state = gtsam.NavState(curr_pose, curr_vel)
        next_state = pim.predict(nav_state, curr_bias)

        curr_pose = next_state.pose()
        curr_vel = next_state.velocity()
        pim.resetIntegration()

        est_poses.append(curr_pose)

    print("Simulaton Finished. Drawing Map...")

    start_lat, start_lon = start_loc
    m_per_deg_lat = 111000.0
    m_per_deg_lon = 111000.0 * np.cos(np.radians(start_lat))

    gt_path = []
    est_path = []

    # [수정] 타입 체크를 통한 좌표 추출
    for p in sim_traj:
        pos = p["pose"].translation()
        if isinstance(pos, np.ndarray):
            px, py = pos[0], pos[1]
        else:
            px, py = pos.x(), pos.y()
        gt_path.append([start_lat + py / m_per_deg_lat, start_lon + px / m_per_deg_lon])

    for p in est_poses:
        pos = p.translation()
        if isinstance(pos, np.ndarray):
            px, py = pos[0], pos[1]
        else:
            px, py = pos.x(), pos.y()
        est_path.append([start_lat + py / m_per_deg_lat, start_lon + px / m_per_deg_lon])

    m = folium.Map(location=start_loc, zoom_start=13)

    folium.PolyLine(gt_path, color="green", weight=5, opacity=0.7, tooltip="Ground Truth").add_to(m)
    folium.PolyLine(
        est_path, color="blue", weight=3, opacity=0.8, tooltip="Proposed (RL+SysID)"
    ).add_to(m)

    folium.Marker(gt_path[0], popup="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(gt_path[-1], popup="End", icon=folium.Icon(color="red")).add_to(m)

    map_file = "long_term_navigation_busan.html"
    m.save(map_file)
    print(f"Map saved to '{map_file}'")

    # 오차 그래프
    gt_xy = np.array(
        [
            (p["pose"].x(), p["pose"].y())
            if not isinstance(p["pose"].translation(), np.ndarray)
            else (p["pose"].translation()[0], p["pose"].translation()[1])
            for p in sim_traj
        ]
    )

    est_xy = np.array(
        [
            (p.x(), p.y())
            if not isinstance(p.translation(), np.ndarray)
            else (p.translation()[0], p.translation()[1])
            for p in est_poses
        ]
    )

    min_len = min(len(gt_xy), len(est_xy))
    error = np.linalg.norm(gt_xy[:min_len] - est_xy[:min_len], axis=1)

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(min_len) * dt / 60, error)
    plt.title("Navigation Position Error over Time (40 mins)")
    plt.xlabel("Time (min)")
    plt.ylabel("Error (m)")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run_navigation_simulation()
