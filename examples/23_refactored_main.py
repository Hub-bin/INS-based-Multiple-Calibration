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
    traj_data = sim.generate_3d_profile(total_duration_min=10)

    # [수정] 군사용 1-mil급 고성능 INS 스펙 적용
    # Gyro Bias: 1e-5 rad/s (approx 2 deg/hr) -> Tactical Grade
    # Accel Bias: 1e-3 m/s^2 (approx 100 ug)
    true_acc = {
        "bias": [1.5e-3, -2.0e-3, 5.0e-4],
        "scale": [1.0001, 0.9999, 1.00005],
        "temp_lin": [1e-5] * 3,
        "temp_non": [1e-7] * 3,
        "hyst": [2e-4, 1e-4, 1e-4],  # 히스테리시스도 매우 작음
    }
    true_gyr = {
        "bias": [1e-5, -1.5e-5, 5e-6],
        "scale": [0.99995, 1.00005, 1.0],
        "temp_lin": [1e-6] * 3,
        "temp_non": [1e-8] * 3,
        "hyst": [1e-6] * 3,
    }

    # Sensor Noise도 고성능에 맞춰 대폭 감소
    imu = ImuSensor(
        accel_bias=true_acc["bias"],
        accel_hysteresis=true_acc["hyst"],
        accel_noise=1e-5,  # Low noise
        gyro_bias=true_gyr["bias"],
        gyro_noise=1e-6,  # Very low noise
    )

    sysid = SysIdCalibrator()
    viz = CalibVisualizer(output_dir=OUTPUT_DIR)

    # 2. Run Modes
    results = {}

    for mode in ["raw", "online"]:
        print(f"\n>>> Running Mode: {mode}")
        nav = StrapdownNavigator(traj_data[0]["pose"], gravity=9.81)

        # Buffers
        h_meas, h_true_acc, h_true_gyr, h_temp = [], [], [], []
        curr_p = None

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
            if data["vel_world"] < 0.05:
                nav.zero_velocity_update()

            # B. Online Calibration
            if mode == "online":
                h_meas.append((meas_acc, meas_gyr))
                h_true_acc.append(data["sf_true"])
                h_true_gyr.append(data["omega_body"])
                h_temp.append(data["temp"])

                if i > 600 and i % 100 == 0:
                    packed_true = list(zip(h_true_acc, h_true_gyr))
                    res = sysid.run(packed_true, h_meas, h_temp)
                    if res:
                        curr_p = res

            # Logging
            if i % 100 == 0:
                log["time"].append(data["time"])
                p = (
                    curr_p
                    if curr_p
                    else {
                        "acc_b": np.zeros(3),
                        "acc_h": np.zeros(3),
                        "acc_k1": np.zeros(3),
                        "acc_k2": np.zeros(3),
                        "acc_T_inv": np.eye(3),
                        "gyr_b": np.zeros(3),
                        "gyr_h": np.zeros(3),
                        "gyr_k1": np.zeros(3),
                        "gyr_k2": np.zeros(3),
                        "gyr_T_inv": np.eye(3),
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
                # Raw 모드
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
