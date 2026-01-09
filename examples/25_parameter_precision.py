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

OUTPUT_DIR = "output_precision"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def run_precision_test():
    # 1. Setup
    dt = 0.1
    road_gen = RoadTrajectoryGenerator(location_point=(35.1796, 129.0756), dist=8000)
    sim = TrajectorySimulator(road_gen, dt)

    # Excitation Profile 사용
    traj_data = sim.generate_excitation_profile(total_duration_min=10)

    # 2. True Parameters (검증용 정답지)
    true_acc = {
        "bias": np.array([0.05, -0.03, 0.02]),
        "scale": np.array([1.02, 0.98, 1.01]),
        "temp_lin": np.array([0.005, 0.005, 0.005]),
        "temp_non": np.array([1e-4, 1e-4, 1e-4]),
        "hyst": np.array([0.01, 0.005, 0.005]),
    }
    true_gyr = {
        "bias": np.array([0.002, -0.002, 0.001]),
        "scale": np.array([0.995, 1.005, 1.0]),
        "temp_lin": np.array([2e-4, 2e-4, 2e-4]),
        "temp_non": np.array([1e-5, 1e-5, 1e-5]),
        "hyst": np.array([1e-3, 1e-3, 1e-3]),
    }

    # [핵심 수정] ImuSensor는 '순수 노이즈'만 생성하도록 설정 (Bias/Hyst=0)
    # 시스템 오차는 아래 루프에서 수동으로 제어하여 주입함
    imu = ImuSensor(
        accel_bias=np.zeros(3),
        accel_hysteresis=np.zeros(3),
        accel_noise=1e-4,  # Noise만 남김
        gyro_bias=np.zeros(3),
        gyro_noise=1e-5,
    )

    sysid = SysIdCalibrator()

    # 3. Buffers
    h_meas, h_true_acc, h_true_gyr, h_temp = [], [], [], []
    log_time = []
    err_log = {
        "acc": {"bias": [], "scale": [], "hyst": []},
        "gyr": {"bias": [], "scale": [], "hyst": []},
    }

    print("\n>>> Running Precision Analysis (Consistency Fixed)...")

    # Hysteresis 생성을 위한 이전 값
    prev_true_acc = np.zeros(3)
    prev_true_gyr = np.zeros(3)

    for i, data in enumerate(traj_data):
        # A. Get Noise (ImuSensor 이용)
        # ImuSensor에 True값과 0 Bias를 넣으면 (True + Noise)가 반환됨
        noisy_acc, noisy_gyr, _ = imu.measure(
            data["pose"], data["sf_true"], data["omega_body"], data["temp"]
        )

        # 순수 노이즈 추출 (Measurement - True)
        noise_acc = noisy_acc - data["sf_true"]
        noise_gyr = noisy_gyr - data["omega_body"]

        # B. Construct Measurement Manually (SysID 모델과 완벽 일치시킴)
        # Model: Meas = (Scale * True) + Bias + Temp + Hyst + Noise

        d_temp = data["temp"] - 20.0

        # --- Accel Construction ---
        acc_diff = data["sf_true"] - prev_true_acc if i > 0 else np.zeros(3)
        acc_hyst_term = true_acc["hyst"] * np.tanh(
            acc_diff * 10.0
        )  # SysID와 동일한 tanh 스케일(10.0) 사용

        sim_acc = (
            (data["sf_true"] * true_acc["scale"])
            + true_acc["bias"]
            + (true_acc["temp_lin"] * d_temp)
            + (true_acc["temp_non"] * (d_temp**2))
            + acc_hyst_term
            + noise_acc
        )

        prev_true_acc = data["sf_true"]

        # --- Gyro Construction ---
        gyr_diff = data["omega_body"] - prev_true_gyr if i > 0 else np.zeros(3)
        gyr_hyst_term = true_gyr["hyst"] * np.tanh(gyr_diff * 10.0)

        sim_gyr = (
            (data["omega_body"] * true_gyr["scale"])
            + true_gyr["bias"]
            + (true_gyr["temp_lin"] * d_temp)
            + (true_gyr["temp_non"] * (d_temp**2))
            + gyr_hyst_term
            + noise_gyr
        )

        prev_true_gyr = data["omega_body"]

        # C. Collect Data
        h_meas.append((sim_acc, sim_gyr))
        h_true_acc.append(data["sf_true"])
        h_true_gyr.append(data["omega_body"])
        h_temp.append(data["temp"])

        # D. Run SysID (Every 1 min)
        if i > 600 and i % 600 == 0:
            packed_true = list(zip(h_true_acc, h_true_gyr))
            mask = np.ones(21)

            res = sysid.run(packed_true, h_meas, h_temp, acc_mask=mask, gyr_mask=mask)

            if res:
                log_time.append(data["time"] / 60.0)

                # Acc Errors
                est_a_scale = 1.0 / np.diag(res["acc_T_inv"])
                err_log["acc"]["bias"].append(res["acc_b"] - true_acc["bias"])
                err_log["acc"]["scale"].append(est_a_scale - true_acc["scale"])
                err_log["acc"]["hyst"].append(res["acc_h"] - true_acc["hyst"])

                # Gyr Errors
                est_g_scale = 1.0 / np.diag(res["gyr_T_inv"])
                err_log["gyr"]["bias"].append(res["gyr_b"] - true_gyr["bias"])
                err_log["gyr"]["scale"].append(est_g_scale - true_gyr["scale"])
                err_log["gyr"]["hyst"].append(res["gyr_h"] - true_gyr["hyst"])

                # Log Norm
                print(
                    f"  [Time {data['time'] / 60:.1f}m] Acc Bias Err: {np.linalg.norm(err_log['acc']['bias'][-1]):.6f} | Scale Err: {np.linalg.norm(err_log['acc']['scale'][-1]):.6f}"
                )

    plot_errors(log_time, err_log)
    print(f"Analysis Complete. Check '{OUTPUT_DIR}/'")


def plot_errors(time, err_log):
    # Accel Errors
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle("Accelerometer Estimation Errors (Est - True)")
    keys = ["bias", "scale", "hyst"]

    for i, key in enumerate(keys):
        data = np.array(err_log["acc"][key])
        if len(data) > 0:
            axes[i].plot(time, data[:, 0], "r-o", label="X")
            axes[i].plot(time, data[:, 1], "g-o", label="Y")
            axes[i].plot(time, data[:, 2], "b-o", label="Z")
            axes[i].axhline(0, color="k", linestyle="--")
            axes[i].set_title(f"{key.capitalize()} Error")
            axes[i].grid(True)
            axes[i].legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/acc_error_analysis.png")

    # Gyro Errors
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle("Gyroscope Estimation Errors (Est - True)")

    for i, key in enumerate(keys):
        data = np.array(err_log["gyr"][key])
        if len(data) > 0:
            axes[i].plot(time, data[:, 0], "r-o", label="X")
            axes[i].plot(time, data[:, 1], "g-o", label="Y")
            axes[i].plot(time, data[:, 2], "b-o", label="Z")
            axes[i].axhline(0, color="k", linestyle="--")
            axes[i].set_title(f"{key.capitalize()} Error")
            axes[i].grid(True)
            axes[i].legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/gyr_error_analysis.png")


if __name__ == "__main__":
    run_precision_test()
