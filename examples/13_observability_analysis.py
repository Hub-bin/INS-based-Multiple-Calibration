import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import gtsam

from src.dynamics.ground import GroundVehicle
from src.sensors.imu import ImuSensor
from src.utils.sensitivity import SensitivityAnalyzer


def generate_scenario_data(scenario_type, dt=0.1, duration=10.0):
    vehicle = GroundVehicle()
    # 순수 물리적 영향(민감도)만 보기 위해 노이즈 제거
    imu = ImuSensor(accel_noise=0.0, gyro_noise=0.0)

    steps = int(duration / dt)
    measurements = []

    vel_x = 0.0

    for i in range(steps):
        t = i * dt

        if scenario_type == "Constant_Velocity":
            vel_x = 10.0
            acc_cmd = 0.0
            yaw_rate = 0.0
        elif scenario_type == "Acceleration":
            vel_x += 2.0 * dt
            acc_cmd = 2.0
            yaw_rate = 0.0
        elif scenario_type == "Turning":
            vel_x = 10.0
            acc_cmd = 0.0
            yaw_rate = 0.5
        elif scenario_type == "Stop_and_Go":
            acc_cmd = 2.0 * np.sin(t)
            vel_x += acc_cmd * dt
            yaw_rate = 0.0

        # True Kinematics
        true_acc = np.array([acc_cmd, yaw_rate * vel_x, 0.0])
        true_gyr = np.array([0.0, 0.0, yaw_rate])

        vehicle.update(dt, vel_x, yaw_rate)
        meas = imu.measure(vehicle.current_pose, true_acc, true_gyr)
        measurements.append(meas)

    return vehicle, measurements


def main():
    print("=== Observability & Sensitivity Analysis (All Parameters) ===")

    scenarios = ["Constant_Velocity", "Acceleration", "Turning", "Stop_and_Go"]

    base_params = {
        "acc_b": np.zeros(3),
        "gyr_b": np.zeros(3),
        "acc_T": np.eye(3),
        "gyr_T": np.eye(3),
    }

    dt = 0.1
    results = {}

    for sc in scenarios:
        print(f"Analyzing Scenario: {sc}...")
        vehicle, meas = generate_scenario_data(sc, dt=dt)
        analyzer = SensitivityAnalyzer(vehicle, None, dt)
        scores = analyzer.compute_sensitivity(meas, base_params)
        results[sc] = scores

    # --- 시각화 (모든 파라미터 포함) ---
    axis_names = ["x", "y", "z"]

    # 1. Bias Keys (6개)
    bias_keys = []
    for sens in ["Acc", "Gyr"]:
        for ax in axis_names:
            bias_keys.append(f"{sens}_Bias_{ax}")

    # 2. Matrix Keys (Scale & Misalign) (18개)
    # T[i, j] -> i: sensor axis, j: true axis
    matrix_keys = []
    for sens in ["Acc", "Gyr"]:
        for i, ax_i in enumerate(axis_names):
            for j, ax_j in enumerate(axis_names):
                # 대각 성분은 Scale, 비대각 성분은 Misalign
                type_name = "Scale" if i == j else "Misalign"
                key_name = f"{sens}_{type_name}_{ax_i}{ax_j}"
                matrix_keys.append(key_name)

    all_keys = bias_keys + matrix_keys  # 총 24개 파라미터

    # 데이터 매트릭스 구성 (Scenario x Parameters)
    data_matrix = []
    for sc in scenarios:
        row = []
        for k in all_keys:
            # 키가 없으면 0.0 처리 (안전장치)
            val = results[sc].get(k, 0.0)
            row.append(val)
        data_matrix.append(row)

    data_matrix = np.array(data_matrix)

    # 시각화 설정
    fig, ax = plt.subplots(figsize=(20, 8))  # 가로로 넓게

    # 민감도 값의 차이가 크므로 로그 스케일 적용 (log(1+x))
    viz_data = np.log1p(data_matrix)

    im = ax.imshow(viz_data, cmap="YlGnBu")

    # 축 설정
    ax.set_xticks(np.arange(len(all_keys)))
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_xticklabels(all_keys, fontsize=10, fontweight="bold")
    ax.set_yticklabels(scenarios, fontsize=12, fontweight="bold")

    # X축 라벨 회전
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Grid 표시 (셀 구분선)
    ax.set_xticks(np.arange(len(all_keys) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(scenarios) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 컬러바 추가
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Sensitivity Score (Log Scale)", rotation=270, labelpad=20)

    ax.set_title("Full Parameter Observability (Sensitivity) Analysis", fontsize=16)

    # 셀 안에 수치 표시 (선택 사항: 값이 너무 많아 복잡할 수 있으므로 주석 처리하거나 필요시 해제)
    # for i in range(len(scenarios)):
    #     for j in range(len(all_keys)):
    #         val = data_matrix[i, j]
    #         if val > 1.0: # 주요 값만 표시
    #             ax.text(j, i, f"{val:.0f}", ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
