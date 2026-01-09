import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import copy

from src.utils.road_generator import RoadTrajectoryGenerator
from src.sensors.imu import ImuSensor
from src.calibration.sysid_corrector import SysIdCalibrator
from src.simulation.profile import TrajectorySimulator
from src.navigation.strapdown import StrapdownNavigator
from src.utils.visualization import CalibVisualizer

# 출력 디렉토리
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
    # 일반 주행 + 3D 거동이 포함된 프로파일 (항법 성능 테스트용)
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

        # Initial State
        init_data = traj_data[0]
        nav = StrapdownNavigator(init_data["pose"], gravity=9.81)
        nav.curr_vel = init_data["vel_world"]

        # Buffers
        h_meas, h_true_acc, h_true_gyr, h_temp = [], [], [], []
        curr_p = None

        # Warm Start States
        last_acc_params = None
        last_gyr_params = None

        # Log
        log = {
            "time": [],
            "acc": {k: [] for k in ["bias", "scale", "temp_lin", "temp_non", "hyst"]},
            "gyr": {k: [] for k in ["bias", "scale", "temp_lin", "temp_non", "hyst"]},
        }

        prev_sf = np.zeros(3)
        prev_omega = np.zeros(3)

        for i, data in enumerate(traj_data):
            # A. Measure
            meas_acc, meas_gyr, _ = imu.measure(
                data["pose"], data["sf_true"], data["omega_body"], data["temp"]
            )

            # Scale Factor Manual Application
            meas_acc = meas_acc * np.array(true_acc["scale"])
            meas_gyr = meas_gyr * np.array(true_gyr["scale"])

            # ZUPT
            if data["speed"] < 0.05:
                nav.zero_velocity_update()

            # B. Online Calibration (Sliding Window + Warm Start)
            if mode == "online":
                h_meas.append((meas_acc, meas_gyr))
                h_true_acc.append(data["sf_true"])
                h_true_gyr.append(data["omega_body"])
                h_temp.append(data["temp"])

                # Window가 꽉 찼고, 업데이트 주기일 때
                if i >= WINDOW_STEPS and i % UPDATE_INTERVAL == 0:
                    # Slicing
                    meas_win = h_meas[-WINDOW_STEPS:]
                    acc_win = h_true_acc[-WINDOW_STEPS:]
                    gyr_win = h_true_gyr[-WINDOW_STEPS:]
                    temp_win = h_temp[-WINDOW_STEPS:]
                    packed_true = list(zip(acc_win, gyr_win))

                    # Boundary Values for Hysteresis
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
                        init_gyr_params=last_gyr_params,  # Warm Start
                        prev_true_acc=prev_t_acc,
                        prev_true_gyr=prev_t_gyr,  # Continuity
                    )

                    if res:
                        curr_p = res
                        # Update Warm Start Guess
                        last_acc_params = res["acc_params"]
                        last_gyr_params = res["gyr_params"]

                        # Logging Status (Optional)
                        if i % 600 == 0:
                            acc_b_err = np.linalg.norm(res["acc_b"] - true_acc["bias"])
                            print(f"  [Time {i * dt / 60:.1f}m] Est BiasErr: {acc_b_err:.6f}")

            # Logging
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

            # C. Correction
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

            # D. Integration
            nav.integrate(corr_acc, corr_gyr, dt)
            nav.predict()

        results[mode] = nav.poses
        if mode == "online":
            viz.plot_params(log, true_acc, true_gyr, dt)

    # 3. Finalize
    print(f"\n[Final Results Check]")
    if curr_p:
        print(f"  > True Acc Bias: {true_acc['bias']}")
        print(f"  > Est  Acc Bias: {curr_p['acc_b']}")

    viz.plot_nav_error(traj_data, results, dt)
    viz.save_map(traj_data, results, start_loc)
    print(f"Simulation Complete. Check '{OUTPUT_DIR}/' for results.")


if __name__ == "__main__":
    run_simulation()
