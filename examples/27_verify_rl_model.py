import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
import gtsam

from src.utils.road_generator import RoadTrajectoryGenerator
from src.sensors.imu import ImuSensor
from src.simulation.profile import TrajectorySimulator
from src.navigation.strapdown import StrapdownNavigator
from src.calibration.rl_agent import PPOAgent

# 설정
MODEL_PATH = "output_high_end/rl_calibrator.pth"
OUTPUT_DIR = "output_verification"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def decode_action(action):
    # [수정] 학습 환경과 동일한 6-Dim Bias Decoding
    return {"acc_bias": action[0:3] * 0.05, "gyr_bias": action[3:6] * 0.005}


def normalize_observation(meas_acc, meas_gyr, temp):
    # [핵심] 학습 환경과 동일한 정규화
    norm_acc = meas_acc / 9.81
    norm_gyr = meas_gyr
    norm_temp = (temp - 20.0) / 30.0
    return np.concatenate([norm_acc, norm_gyr, [norm_temp]])


def run_verification():
    print(f"\n>>> Loading RL Model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("Model not found.")
        return

    # [수정] action_dim=6
    agent = PPOAgent(state_dim=(600, 7), action_dim=6)
    try:
        agent.policy.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    except:
        return
    agent.policy.eval()

    dt = 0.1
    start_loc = (35.1796, 129.0756)
    road_gen = RoadTrajectoryGenerator(location_point=start_loc, dist=8000)
    sim = TrajectorySimulator(road_gen, dt)
    traj_data = sim.generate_3d_profile(total_duration_min=10)

    imu = ImuSensor(
        accel_bias=np.zeros(3),
        accel_hysteresis=np.zeros(3),
        accel_noise=1e-5,
        gyro_bias=np.zeros(3),
        gyro_noise=1e-6,
    )

    # [검증용 온도 계수] 학습 환경과 동일하게 설정 (Fixed Physics)
    temp_coeffs = {
        "acc_lin": np.array([0.001, 0.001, 0.001]),
        "acc_quad": np.array([0.0, 0.0, 0.0]),
        "gyr_lin": np.array([0.00001, 0.00001, 0.00001]),
    }

    nav_raw = StrapdownNavigator(traj_data[0]["pose"], gravity=9.81)
    nav_rl = StrapdownNavigator(traj_data[0]["pose"], gravity=9.81)
    nav_raw.curr_vel = traj_data[0]["vel_world"]
    nav_rl.curr_vel = traj_data[0]["vel_world"]

    obs_buffer = []
    WINDOW_SIZE = 600
    traj_gt, traj_raw, traj_rl = [], [], []
    curr_bias = {"acc_bias": np.zeros(3), "gyr_bias": np.zeros(3)}

    print(">>> Running Verification Loop (Bias Only)...")

    for i, data in enumerate(traj_data):
        p = data["pose"].translation()
        traj_gt.append([p.x(), p.y(), p.z()] if hasattr(p, "x") else p)

        meas_acc, meas_gyr, _ = imu.measure(
            data["pose"], data["sf_true"], data["omega_body"], data["temp"]
        )

        # [시나리오] 학습 환경과 동일한 오차 주입
        dt_temp = data["temp"] - 20.0
        meas_acc += temp_coeffs["acc_lin"] * dt_temp
        meas_gyr += temp_coeffs["gyr_lin"] * dt_temp

        if data["speed"] < 0.05:
            nav_raw.zero_velocity_update()
            nav_rl.zero_velocity_update()

        # [A] Raw
        nav_raw.integrate(meas_acc, meas_gyr, dt)
        pose_raw = nav_raw.predict()
        pr = pose_raw.translation()
        traj_raw.append([pr.x(), pr.y(), pr.z()] if hasattr(pr, "x") else pr)

        # [B] RL
        obs_row = normalize_observation(meas_acc, meas_gyr, data["temp"])
        obs_buffer.append(obs_row)

        if len(obs_buffer) >= WINDOW_SIZE:
            state = np.array(obs_buffer[-WINDOW_SIZE:], dtype=np.float32)
            if i % 10 == 0:
                action, _ = agent.select_action(state)
                curr_bias = decode_action(action)
            if len(obs_buffer) > WINDOW_SIZE * 2:
                obs_buffer = obs_buffer[-WINDOW_SIZE:]

        corr_acc_rl = meas_acc - curr_bias["acc_bias"]
        corr_gyr_rl = meas_gyr - curr_bias["gyr_bias"]

        nav_rl.integrate(corr_acc_rl, corr_gyr_rl, dt)
        pose_rl = nav_rl.predict()
        pl = pose_rl.translation()
        traj_rl.append([pl.x(), pl.y(), pl.z()] if hasattr(pl, "x") else pl)

    traj_gt = np.array(traj_gt)
    traj_raw = np.array(traj_raw)
    traj_rl = np.array(traj_rl)

    err_raw = np.linalg.norm(traj_gt - traj_raw, axis=1)
    err_rl = np.linalg.norm(traj_gt - traj_rl, axis=1)

    print(f"\n[Final Results (Bias Only)]")
    print(f"  > Mean Error (Raw): {np.mean(err_raw):.4f} m")
    print(f"  > Mean Error (RL) : {np.mean(err_rl):.4f} m")

    plt.figure(figsize=(10, 6))
    plt.plot(err_raw, "r--", label="Raw (Temp Drift)")
    plt.plot(err_rl, "b-", label="AI Agent")
    plt.title("Navigation Error Comparison (Bias Only)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/verification_result.png")


if __name__ == "__main__":
    run_verification()
