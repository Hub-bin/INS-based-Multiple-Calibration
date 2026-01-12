import sys
import os

# 프로젝트 루트 경로를 확실하게 sys.path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import copy
import torch

from src.utils.road_generator import RoadTrajectoryGenerator
from src.sensors.imu import ImuSensor
from src.calibration.sysid_corrector import SysIdCalibrator
from src.simulation.profile import TrajectorySimulator
from src.navigation.strapdown import StrapdownNavigator
from src.utils.visualization import CalibVisualizer
from src.calibration.rl_agent import PPOAgent

# 출력 디렉토리 정의
OUTPUT_DIR = "output_high_end"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def run_simulation():
    # 1. Setup
    start_loc = (35.1796, 129.0756)
    dt = 0.1
    road_gen = RoadTrajectoryGenerator(location_point=start_loc, dist=8000)

    # Simulator
    sim = TrajectorySimulator(road_gen, dt)
    traj_data = sim.generate_3d_profile(total_duration_min=10)

    # [설정] 1-mil급 고성능 INS 스펙 (Tactical Grade)
    true_acc = {
        "bias": [1.5e-3, -2.0e-3, 5.0e-4],
        "scale": [1.0001, 0.9999, 1.00005],
        "temp_lin": [1e-5] * 3,
        "temp_non": [1e-7] * 3,
        "hyst": [2e-4, 1e-4, 1e-4],
    }
    true_gyr = {
        "bias": [1e-5, -1.5e-5, 5e-6],
        "scale": [0.99995, 1.00005, 1.0],
        "temp_lin": [1e-6] * 3,
        "temp_non": [1e-8] * 3,
        "hyst": [1e-6] * 3,
    }

    # Noise도 대폭 감소
    imu = ImuSensor(
        accel_bias=true_acc["bias"],
        accel_hysteresis=true_acc["hyst"],
        accel_noise=1e-5,
        gyro_bias=true_gyr["bias"],
        gyro_noise=1e-6,
    )

    sysid = SysIdCalibrator()
    viz = CalibVisualizer(output_dir=OUTPUT_DIR)

    # 2. Run Modes
    results = {}

    # Sliding Window 설정
    WINDOW_SEC = 60.0
    WINDOW_STEPS = int(WINDOW_SEC / dt)
    UPDATE_INTERVAL = int(10.0 / dt)

    for mode in ["raw", "online"]:
        print(f"\n>>> Running Mode: {mode}")

        init_data = traj_data[0]
        nav = StrapdownNavigator(init_data["pose"], gravity=9.81)
        nav.curr_vel = init_data["vel_world"]

        h_meas, h_true_acc, h_true_gyr, h_temp = [], [], [], []
        curr_p = None
        last_acc_params = None
        last_gyr_params = None

        log = {
            "time": [],
            "acc": {k: [] for k in ["bias", "scale", "temp_lin", "temp_non", "hyst"]},
            "gyr": {k: [] for k in ["bias", "scale", "temp_lin", "temp_non", "hyst"]},
        }

        prev_sf = np.zeros(3)
        prev_omega = np.zeros(3)

        for i, data in enumerate(traj_data):
            meas_acc, meas_gyr, _ = imu.measure(
                data["pose"], data["sf_true"], data["omega_body"], data["temp"]
            )

            meas_acc = meas_acc * np.array(true_acc["scale"])
            meas_gyr = meas_gyr * np.array(true_gyr["scale"])

            if data["speed"] < 0.05:
                nav.zero_velocity_update()

            if mode == "online":
                h_meas.append((meas_acc, meas_gyr))
                h_true_acc.append(data["sf_true"])
                h_true_gyr.append(data["omega_body"])
                h_temp.append(data["temp"])

                if i >= WINDOW_STEPS and i % UPDATE_INTERVAL == 0:
                    meas_win = h_meas[-WINDOW_STEPS:]
                    acc_win = h_true_acc[-WINDOW_STEPS:]
                    gyr_win = h_true_gyr[-WINDOW_STEPS:]
                    temp_win = h_temp[-WINDOW_STEPS:]
                    packed_true = list(zip(acc_win, gyr_win))

                    prev_t_acc = h_true_acc[-WINDOW_STEPS - 1]
                    prev_t_gyr = h_true_gyr[-WINDOW_STEPS - 1]

                    mask = np.ones(21)
                    res = sysid.run(
                        packed_true,
                        meas_win,
                        temp_win,
                        acc_mask=mask,
                        gyr_mask=mask,
                        init_acc_params=last_acc_params,
                        init_gyr_params=last_gyr_params,
                        prev_true_acc=prev_t_acc,
                        prev_true_gyr=prev_t_gyr,
                    )

                    if res:
                        curr_p = res
                        last_acc_params = res["acc_params"]
                        last_gyr_params = res["gyr_params"]

                        if i % 600 == 0:
                            acc_b_err = np.linalg.norm(res["acc_b"] - true_acc["bias"])
                            print(f"  [Time {i * dt / 60:.1f}m] Est BiasErr: {acc_b_err:.6f}")

            if i % 100 == 0:
                log["time"].append(data["time"])
                p = (
                    curr_p
                    if curr_p
                    else {
                        k: np.zeros(3) if "inv" not in k else np.eye(3)
                        for k in [
                            "acc_b",
                            "acc_h",
                            "acc_k1",
                            "acc_k2",
                            "acc_T_inv",
                            "gyr_b",
                            "gyr_h",
                            "gyr_k1",
                            "gyr_k2",
                            "gyr_T_inv",
                        ]
                    }
                )

                for s in ["acc", "gyr"]:
                    log[s]["bias"].append(p[f"{s}_b"])
                    log[s]["scale"].append(np.diag(p[f"{s}_T_inv"]))
                    log[s]["hyst"].append(p[f"{s}_h"])
                    log[s]["temp_lin"].append(p[f"{s}_k1"])
                    log[s]["temp_non"].append(p[f"{s}_k2"])

            if mode == "raw" or curr_p is None:
                corr_acc = meas_acc
                corr_gyr = meas_gyr
            else:
                p = curr_p
                # Acc Correction
                diff = data["sf_true"] - prev_sf
                h_sign = np.tanh(diff * 10.0)
                dt_t = data["temp"] - 20.0

                acc_err = (
                    p["acc_b"] + p["acc_k1"] * dt_t + p["acc_k2"] * dt_t**2 + p["acc_h"] * h_sign
                )
                corr_acc = p["acc_T_inv"] @ (meas_acc - acc_err)

                # Gyr Correction
                g_diff = data["omega_body"] - prev_omega
                gh_sign = np.tanh(g_diff * 10.0)
                g_err = (
                    p["gyr_b"] + p["gyr_k1"] * dt_t + p["gyr_k2"] * dt_t**2 + p["gyr_h"] * gh_sign
                )
                corr_gyr = p["gyr_T_inv"] @ (meas_gyr - g_err)

            prev_sf = data["sf_true"]
            prev_omega = data["omega_body"]

            nav.integrate(corr_acc, corr_gyr, dt)
            nav.predict()

        results[mode] = nav.poses
        if mode == "online":
            viz.plot_params(log, true_acc, true_gyr, dt)

    print(f"\n[Final Results Check]")
    if curr_p:
        print(f"  > True Acc Bias: {true_acc['bias']}")
        print(f"  > Est  Acc Bias: {curr_p['acc_b']}")

    viz.plot_nav_error(traj_data, results, dt)
    viz.save_map(traj_data, results, start_loc)
    print(f"Simulation Complete. Check '{OUTPUT_DIR}/' for results.")


def run_rl_training():
    print("\n>>> Starting RL Calibration Training (Bias Only)...")
    from src.calibration.calibration_env import CalibrationEnv

    start_loc = (35.1796, 129.0756)
    road_gen = RoadTrajectoryGenerator(location_point=start_loc, dist=8000)

    # 윈도우 크기 600 (60초)
    env = CalibrationEnv(road_gen, dt=0.1, window_size=600)

    # [수정] Action Dim 18 -> 6 (Bias Only)
    agent = PPOAgent(state_dim=(600, 7), action_dim=6)

    MAX_EPISODES = 300  # 300회 정도면 Bias 패턴 파악 가능

    for episode in range(MAX_EPISODES):
        state, _ = env.reset()
        done = False
        total_reward = 0
        buffer = []

        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.append((state, action, log_prob, reward, done))
            state = next_state
            total_reward += reward

        agent.update(buffer)
        print(f"Episode {episode + 1}/{MAX_EPISODES} - Total Reward: {total_reward:.4f}")

    torch.save(agent.policy.state_dict(), f"{OUTPUT_DIR}/rl_calibrator.pth")
    print("Training Complete. Model Saved.")


if __name__ == "__main__":
    print("Select Mode:")
    print("1: Standard Simulation (Optimization based)")
    print("2: RL Training (PPO based)")

    try:
        mode = input("Enter mode (1 or 2): ").strip()
    except EOFError:
        mode = "1"

    if mode == "2":
        run_rl_training()
    else:
        run_simulation()
