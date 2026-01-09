import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import gtsam

from src.utils.road_generator import RoadTrajectoryGenerator
from src.sensors.imu import ImuSensor
from src.calibration.sysid_corrector import SysIdCalibrator
from src.simulation.profile import TrajectorySimulator

OUTPUT_DIR = "output_sliding"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def run_sliding_window_test():
    dt = 0.1
    WINDOW_SEC = 60.0
    WINDOW_STEPS = int(WINDOW_SEC / dt)
    UPDATE_INTERVAL = int(10.0 / dt)

    road_gen = RoadTrajectoryGenerator(location_point=(35.1796, 129.0756), dist=8000)
    sim = TrajectorySimulator(road_gen, dt)
    traj_data = sim.generate_excitation_profile(total_duration_min=10)

    true_acc = {
        "bias": [0.05, -0.03, 0.02],
        "scale": [1.02, 0.98, 1.01],
        "temp_lin": [0.005] * 3,
        "temp_non": [1e-4] * 3,
        "hyst": [0.01, 0.005, 0.005],
    }
    true_gyr = {
        "bias": [0.002, -0.002, 0.001],
        "scale": [0.995, 1.005, 1.0],
        "temp_lin": [2e-4] * 3,
        "temp_non": [1e-5] * 3,
        "hyst": [1e-3] * 3,
    }

    imu = ImuSensor(
        accel_bias=np.zeros(3),
        accel_hysteresis=np.zeros(3),
        accel_noise=1e-4,
        gyro_bias=np.zeros(3),
        gyro_noise=1e-5,
    )
    sysid = SysIdCalibrator()

    h_meas, h_true_acc, h_true_gyr, h_temp = [], [], [], []
    log_time = []
    err_log = {
        "acc": {"bias": [], "scale": [], "hyst": []},
        "gyr": {"bias": [], "scale": [], "hyst": []},
    }

    # [핵심] Warm Start를 위한 파라미터 저장 변수
    last_acc_params = None
    last_gyr_params = None

    print(f"\n>>> Running Sliding Window Test with Warm Start (Window: {WINDOW_SEC}s)...")

    prev_true_acc = np.zeros(3)
    prev_true_gyr = np.zeros(3)

    for i, data in enumerate(traj_data):
        # A. Measure & Construct (Twin Experiment)
        noisy_acc, noisy_gyr, _ = imu.measure(
            data["pose"], data["sf_true"], data["omega_body"], data["temp"]
        )
        noise_acc = noisy_acc - data["sf_true"]
        noise_gyr = noisy_gyr - data["omega_body"]
        d_temp = data["temp"] - 20.0

        acc_diff = data["sf_true"] - prev_true_acc if i > 0 else np.zeros(3)
        acc_hyst = np.array(true_acc["hyst"]) * np.tanh(acc_diff * 10.0)
        sim_acc = (
            (data["sf_true"] * true_acc["scale"])
            + true_acc["bias"]
            + (np.array(true_acc["temp_lin"]) * d_temp)
            + (np.array(true_acc["temp_non"]) * d_temp**2)
            + acc_hyst
            + noise_acc
        )

        gyr_diff = data["omega_body"] - prev_true_gyr if i > 0 else np.zeros(3)
        gyr_hyst = np.array(true_gyr["hyst"]) * np.tanh(gyr_diff * 10.0)
        sim_gyr = (
            (data["omega_body"] * true_gyr["scale"])
            + true_gyr["bias"]
            + (np.array(true_gyr["temp_lin"]) * d_temp)
            + (np.array(true_gyr["temp_non"]) * d_temp**2)
            + gyr_hyst
            + noise_gyr
        )

        prev_true_acc = data["sf_true"]
        prev_true_gyr = data["omega_body"]

        h_meas.append((sim_acc, sim_gyr))
        h_true_acc.append(data["sf_true"])
        h_true_gyr.append(data["omega_body"])
        h_temp.append(data["temp"])

        # B. Run SysID (Sliding Window)
        if i >= WINDOW_STEPS and i % UPDATE_INTERVAL == 0:
            meas_window = h_meas[-WINDOW_STEPS:]
            acc_window = h_true_acc[-WINDOW_STEPS:]
            gyr_window = h_true_gyr[-WINDOW_STEPS:]
            temp_window = h_temp[-WINDOW_STEPS:]
            packed_true = list(zip(acc_window, gyr_window))
            mask = np.ones(21)

            # [핵심] 윈도우 바로 직전의 참값 (Boundary Continuity)
            p_t_acc = h_true_acc[-WINDOW_STEPS - 1]
            p_t_gyr = h_true_gyr[-WINDOW_STEPS - 1]

            res = sysid.run(
                packed_true,
                meas_window,
                temp_window,
                acc_mask=mask,
                gyr_mask=mask,
                init_acc_params=last_acc_params,
                init_gyr_params=last_gyr_params,  # Warm Start
                prev_true_acc=p_t_acc,
                prev_true_gyr=p_t_gyr,  # History
            )

            if res:
                # Update warm start params
                last_acc_params = res["acc_params"]
                last_gyr_params = res["gyr_params"]

                log_time.append(data["time"] / 60.0)

                est_a_scale = 1.0 / np.diag(res["acc_T_inv"])
                err_log["acc"]["bias"].append(res["acc_b"] - true_acc["bias"])
                err_log["acc"]["scale"].append(est_a_scale - true_acc["scale"])
                err_log["acc"]["hyst"].append(res["acc_h"] - true_acc["hyst"])

                est_g_scale = 1.0 / np.diag(res["gyr_T_inv"])
                err_log["gyr"]["bias"].append(res["gyr_b"] - true_gyr["bias"])
                err_log["gyr"]["scale"].append(est_g_scale - true_gyr["scale"])
                err_log["gyr"]["hyst"].append(res["gyr_h"] - true_gyr["hyst"])

                if i % 600 == 0:
                    a_b_err = np.linalg.norm(err_log["acc"]["bias"][-1])
                    a_s_err = np.linalg.norm(err_log["acc"]["scale"][-1])
                    print(
                        f"  [Time {data['time'] / 60:.1f}m] Acc BiasErr: {a_b_err:.6f} | ScaleErr: {a_s_err:.6f}"
                    )

    plot_errors(log_time, err_log)
    print(f"Test Complete. Results saved in '{OUTPUT_DIR}/'")


def plot_errors(time, err_log):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle("Sliding Window Estimation Errors (Warm Start)")
    keys = ["bias", "scale", "hyst"]
    for i, key in enumerate(keys):
        data = np.array(err_log["acc"][key])
        if len(data) > 0:
            axes[i].plot(time, data[:, 0], "r-", label="X")
            axes[i].plot(time, data[:, 1], "g-", label="Y")
            axes[i].plot(time, data[:, 2], "b-", label="Z")
            axes[i].axhline(0, color="k", linestyle="--")
            axes[i].set_title(f"Accel {key.capitalize()} Error")
            axes[i].grid(True)
            if i == 0:
                axes[i].legend()
    axes[2].set_xlabel("Time (min)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sliding_warm_start.png")


if __name__ == "__main__":
    run_sliding_window_test()
