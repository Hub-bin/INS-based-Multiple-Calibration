import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import gtsam

from src.sensors.imu import ImuSensor


def main():
    print("=== Hysteresis Effect Simulation ===")

    # 1. 센서 설정
    # 노이즈를 아주 작게 하여 루프 형상을 명확히 확인
    imu = ImuSensor(
        accel_noise=0.01,
        gyro_noise=0.0,
        accel_bias=np.zeros(3),
        accel_hysteresis=0.2,  # 0.2 m/s^2 정도의 히스테리시스 오차
    )

    # 2. 입력 데이터 생성 (Sine Wave)
    # -10 ~ +10 m/s^2 범위를 왕복
    dt = 0.05
    duration = 20.0
    steps = int(duration / dt)
    t = np.arange(steps) * dt

    # 입력 가속도 (True Input)
    input_acc_x = 10.0 * np.sin(0.5 * t)

    measured_acc_x = []
    hysteresis_error = []

    pose = gtsam.Pose3()  # 고정 위치

    print("Simulating cyclic acceleration...")
    for i in range(steps):
        # X축으로만 가속도 입력
        true_acc = np.array([input_acc_x[i], 0.0, 0.0])
        true_gyr = np.zeros(3)

        # 중력 제거를 위해, measure 함수 내부의 unrotate(g)를 고려
        # measure 내부: specific_force = true_acc_body - g_body
        # 여기선 간단히 true_acc가 specific force라고 가정하고 싶음.
        # 따라서 measure에 넣을 때 g_body를 상쇄하도록 넣어주거나,
        # ImuSensor를 수정하지 않고 '중력이 없는 우주 공간'이라고 가정 (gravity_world=0 설정 불가하므로)
        # -> ImuSensor measure는 pose를 이용해 g를 뺌.
        # Pose가 Identity이므로 g_body = [0, 0, -9.81].
        # 따라서 true_acc_body 입력에 [ax, 0, -9.81]을 넣으면
        # Specific Force = [ax, 0, 0]이 됨.

        input_vec = np.array([input_acc_x[i], 0.0, -9.81])

        meas_acc, meas_gyr, biases = imu.measure(pose, input_vec, true_gyr)

        measured_acc_x.append(meas_acc[0])

        # Bias 항에 포함된 Hysteresis 성분 추출 (Base Bias가 0이므로 Bias 전체가 Hysteresis+Temp)
        # Temp는 0이므로 Bias 전체가 Hysteresis
        hysteresis_error.append(biases[0][0])

    # 3. 시각화
    plt.figure(figsize=(12, 5))

    # Time Series
    plt.subplot(1, 2, 1)
    plt.plot(t, input_acc_x, "k--", label="Input (True)")
    plt.plot(t, measured_acc_x, "b-", alpha=0.6, label="Measured (Hysteresis)")
    plt.title("Acceleration Time Series")
    plt.xlabel("Time (s)")
    plt.ylabel("Accel X (m/s^2)")
    plt.legend()
    plt.grid(True)

    # Hysteresis Loop (Input vs Output)
    plt.subplot(1, 2, 2)
    plt.plot(input_acc_x, measured_acc_x, "b-", label="Input-Output Loop")

    # 오차만 따로 그려서 루프 확인 (Input vs Error)
    # plt.plot(input_acc_x, hysteresis_error, 'r-', label='Hysteresis Error Loop')

    plt.title("Hysteresis Loop (Input vs Output)")
    plt.xlabel("Input Acceleration (m/s^2)")
    plt.ylabel("Measured Acceleration (m/s^2)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("Check the Hysteresis Loop graph.")
    print("If it shows a loop instead of a single line, Hysteresis is working.")


if __name__ == "__main__":
    main()
