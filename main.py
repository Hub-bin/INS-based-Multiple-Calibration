import matplotlib.pyplot as plt
import numpy as np
import gtsam

from src.dynamics.ground import GroundVehicle
from src.sensors.camera import CameraSensor
from src.calibration.hand_eye import HandEyeCalibrator


def main():
    print("--- IMU(Body)-Camera Hand-Eye Calibration Simulation ---")

    # 1. 설정
    dt = 0.1
    sim_duration = 20.0

    vehicle = GroundVehicle()
    # 카메라 센서 (Body 기준 위치: x=2.0, z=1.0)
    # 실제로는 우리가 sensors/camera.py에서 정의한 body_to_camera 값을 찾아내야 함
    cam_sensor = CameraSensor()
    true_T_bc = cam_sensor.body_to_camera

    print(f"True Extrinsics (Body -> Camera):")
    print(true_T_bc)

    # 2. 데이터 수집 (궤적 생성)
    # Hand-Eye Calibration은 충분한 회전이 있어야 잘 됨 (S자 주행)
    velocity_x = 5.0

    body_poses = []  # T_w_b (from GPS/IMU Integration)
    camera_poses = []  # T_w_c (from Visual Odometry / SLAM)

    print("Simulating Trajectory (S-Curve)...")
    steps = int(sim_duration / dt)
    for i in range(steps):
        # S자 주행: Yaw Rate를 사인파로 변경
        yaw_rate = 0.5 * np.sin(i * dt * 0.5)

        vehicle.update(dt, velocity_x, yaw_rate)

        # 현재 Body Pose (Ground Truth) -> 실제론 IMU/GPS로 추정된 값
        T_wb = vehicle.current_pose

        # 현재 Camera Pose (Ground Truth) -> 실제론 Visual Odometry로 추정된 값
        # T_wc = T_wb * T_bc
        T_wc = T_wb.compose(true_T_bc)

        # 노이즈 추가 (Visual Odometry가 GPS보다 보통 부정확하므로 노이즈 섞음)
        # 위치 10cm, 회전 0.05rad 정도 흔들림
        noise = gtsam.Pose3(
            gtsam.Rot3.Ypr(np.random.normal(0, 0.01), 0, 0),
            gtsam.Point3(
                np.random.normal(0, 0.05), np.random.normal(0, 0.05), np.random.normal(0, 0.05)
            ),
        )
        T_wc_noisy = T_wc.compose(noise)

        body_poses.append(T_wb)
        camera_poses.append(T_wc_noisy)

    print(f"Collected {len(body_poses)} pose pairs.")

    # 3. 초기값 설정 (Perturbation)
    # 정답에서 위치 0.5m, 회전 0.2rad 정도 틀어지게 만듦 (꽤 큰 오차)
    perturbation = gtsam.Pose3(gtsam.Rot3.Ypr(0.2, 0.1, -0.1), gtsam.Point3(0.5, -0.3, 0.2))
    initial_guess = true_T_bc.compose(perturbation)

    print("-" * 30)
    print("Initial Guess (Perturbed):")
    print(initial_guess)

    # 4. Calibration 실행
    calibrator = HandEyeCalibrator()
    optimized_T_bc = calibrator.run(body_poses, camera_poses, initial_guess)

    # 5. 결과 비교
    print("-" * 30)
    print("Optimized Result:")
    print(optimized_T_bc)

    # 오차 계산
    error_pose = true_T_bc.between(optimized_T_bc)
    trans_err = np.linalg.norm(error_pose.translation())
    rot_err = np.linalg.norm(error_pose.rotation().xyz())

    print("-" * 30)
    print(f"Estimation Error:")
    print(f"  Translation: {trans_err:.6f} m")
    print(f"  Rotation   : {rot_err:.6f} rad")

    if trans_err < 0.1 and rot_err < 0.1:
        print(">> Calibration SUCCESS!")
    else:
        print(">> Calibration WARNING!")


if __name__ == "__main__":
    main()
