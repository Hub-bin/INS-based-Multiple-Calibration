import sys
import os

# 부모 디렉토리를 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import gtsam

from src.sensors.camera import CameraSensor
from src.sensors.lidar import LidarSensor
from src.calibration.extrinsics import LidarCameraCalibrator


def main():
    print("--- LiDAR-Camera Extrinsic Calibration Simulation ---")

    # 1. 센서 설정 (Truth)
    cam_sensor = CameraSensor(width=640, height=480, fov=90.0, noise_sigma=1.0)

    # 2. True Extrinsics 정의
    vehicle_pose = gtsam.Pose3()  # (0,0,0)

    pose_cam_world = vehicle_pose.compose(cam_sensor.body_to_camera)
    pose_lid_world = vehicle_pose.compose(gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0, 0, 1.5)))

    # T_lid_cam (LiDAR 프레임 기준 카메라의 Pose)
    true_pose_cam_in_lidar = pose_lid_world.inverse().compose(pose_cam_world)

    print(f"True Extrinsics (Camera Pose in LiDAR Frame):")
    print(true_pose_cam_in_lidar)
    print("-" * 30)

    # 3. 데이터 생성
    correspondences = []
    valid_count = 0

    print("Generating Correspondences...")
    for _ in range(200):
        lx = np.random.uniform(5, 25)
        ly = np.random.uniform(-10, 10)
        lz = np.random.uniform(-3, 3)
        pt_lidar = gtsam.Point3(lx, ly, lz)

        camera = gtsam.PinholeCameraCal3_S2(true_pose_cam_in_lidar, cam_sensor.calibration)

        try:
            uv = camera.project(pt_lidar)
            if (0 <= uv[0] < cam_sensor.width) and (0 <= uv[1] < cam_sensor.height):
                u_noise = np.random.normal(0, 0.5)
                v_noise = np.random.normal(0, 0.5)
                measured_uv = gtsam.Point2(uv[0] + u_noise, uv[1] + v_noise)

                # Numpy Array로 저장 (extrinsics.py 수정사항 반영)
                correspondences.append((np.array([lx, ly, lz]), measured_uv))
                valid_count += 1
        except:
            pass

    print(f"  -> Generated {valid_count} valid pairs.")

    # 4. 초기값 설정 (Perturbation)
    noise_pose = gtsam.Pose3(gtsam.Rot3.Ypr(0.1, 0.05, -0.05), gtsam.Point3(0.2, -0.1, 0.1))
    initial_guess = true_pose_cam_in_lidar.compose(noise_pose)

    print("-" * 30)
    print("Initial Guess (Perturbed):")
    print(initial_guess)

    # 5. Calibration 실행
    calibrator = LidarCameraCalibrator(cam_sensor.calibration)
    optimized_pose = calibrator.run(correspondences, initial_guess)

    # 6. 결과 비교
    print("-" * 30)
    print("Optimized Result:")
    print(optimized_pose)

    error_pose = true_pose_cam_in_lidar.between(optimized_pose)
    translation_error = np.linalg.norm(error_pose.translation())
    rotation_error = np.linalg.norm(error_pose.rotation().xyz())

    print("-" * 30)
    print(f"Estimation Error:")
    print(f"  Translation: {translation_error:.6f} m")
    print(f"  Rotation   : {rotation_error:.6f} rad")

    if translation_error < 0.05 and rotation_error < 0.05:
        print(">> Calibration SUCCESS! (Errors are small)")
    else:
        print(">> Calibration WARNING! (Errors are large)")


if __name__ == "__main__":
    main()
