import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import gtsam

from src.dynamics.ground import GroundVehicle
from src.sensors.imu import ImuSensor


def main():
    print("=== Temperature Dependent Bias Drift Simulation ===")

    dt = 0.1
    duration = 60.0  # 1분간 시뮬레이션
    steps = int(duration / dt)

    # 1. 센서 설정 (온도 민감도 추가)
    # 가속도: 온도 1도당 0.01 m/s^2 변함 (꽤 큼)
    # 자이로: 온도 1도당 0.001 rad/s 변함
    # 비선형성: 제곱항 계수 추가
    imu = ImuSensor(
        accel_noise=0.005,
        gyro_noise=0.0005,
        accel_bias=[0.0, 0.0, 0.0],
        gyro_bias=[0.0, 0.0, 0.0],
        ref_temperature=20.0,
        accel_temp_coeff_linear=0.01,
        accel_temp_coeff_nonlinear=0.0005,
        gyro_temp_coeff_linear=0.001,
    )

    vehicle = GroundVehicle()

    # 데이터 저장소
    temps = []
    true_acc_biases = []
    measured_acc_x = []
    positions = []

    # 항법 초기화 (Zero Bias 가정)
    pim_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
    zero_bias = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
    pim = gtsam.PreintegratedImuMeasurements(pim_params, zero_bias)

    curr_pose = gtsam.Pose3()
    curr_vel = gtsam.Point3(0, 0, 0)

    print("Simulating Self-heating (20C -> 60C)...")

    for i in range(steps):
        t = i * dt

        # 2. 온도 모델링 (지수 함수적으로 상승 가정 or 선형 상승)
        # 20도에서 시작해서 60도까지 오름
        current_temp = 20.0 + (40.0 * (t / duration))

        # 3. 차량 상태 (정지 상태 유지)
        vehicle.update(dt, 0.0, 0.0)

        # 4. 측정 (온도 입력)
        # 정지 상태이므로 True Accel = 0, True Gyro = 0
        meas_acc, meas_gyr, true_bias = imu.measure(
            vehicle.current_pose, np.zeros(3), np.zeros(3), temperature=current_temp
        )

        # 5. 항법 수행 (온도 보정 없이 적분 -> Drift 발생)
        pim.integrateMeasurement(meas_acc, meas_gyr, dt)
        nav_state = gtsam.NavState(curr_pose, curr_vel)
        next_state = pim.predict(nav_state, zero_bias)

        curr_pose = next_state.pose()
        curr_vel = next_state.velocity()

        # Reset integration
        pim.resetIntegration()

        # 저장
        temps.append(current_temp)
        true_acc_biases.append(true_bias[0][0])  # Accel X Bias True
        measured_acc_x.append(meas_acc[0])  # Accel X Meas
        positions.append(curr_pose.translation())

    # --- 시각화 ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # 1. 온도 변화
    ax1.plot(np.arange(steps) * dt, temps, "r-", linewidth=2)
    ax1.set_title("IMU Temperature Profile")
    ax1.set_ylabel("Temperature (°C)")
    ax1.grid(True)

    # 2. Bias Drift
    #
    # 온도가 오름에 따라 바이어스가 변하는 모습
    ax2.plot(temps, true_acc_biases, "b-", label="True Bias (Drift)", linewidth=2)
    ax2.scatter(temps[::10], measured_acc_x[::10], color="gray", s=5, alpha=0.3, label="Noisy Meas")
    ax2.set_title("Accel Bias Drift vs Temperature")
    ax2.set_xlabel("Temperature (°C)")
    ax2.set_ylabel("Accel Bias X (m/s²)")
    ax2.legend()
    ax2.grid(True)

    # 3. Position Error (Drift)
    pos_err = [np.linalg.norm(p) for p in positions]  # 정지 상태이므로 위치 자체가 에러
    ax3.plot(np.arange(steps) * dt, pos_err, "k-", linewidth=2)
    ax3.set_title("Position Drift due to Uncompensated Temp Effect")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Position Error (m)")
    ax3.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
