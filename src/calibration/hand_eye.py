import gtsam
import numpy as np


class HandEyeCalibrator:
    """
    IMU(Body)-Camera Extrinsic Calibration 모듈 (Hand-Eye Calibration)
    - 차량이 움직이면서 수집된 Body Pose(from IMU/GPS)와 Camera Pose(from Visual Odometry)를 비교하여
    - Body 프레임과 Camera 프레임 사이의 고정된 변환 행렬(Extrinsics)을 추정합니다.
    """

    def __init__(self):
        pass

    def run(self, body_poses, camera_poses, initial_guess_bc: gtsam.Pose3):
        """
        :param body_poses: List of gtsam.Pose3 (World -> Body 궤적)
        :param camera_poses: List of gtsam.Pose3 (World -> Camera 궤적)
        :param initial_guess_bc: 초기 추정값 (Body -> Camera 변환 행렬)
        :return: optimized T_body_camera
        """
        graph = gtsam.NonlinearFactorGraph()
        initial_estimates = gtsam.Values()

        # 최적화할 변수 Key: T_bc (Extrinsics) - 하나만 존재 (Static)
        CALIB_KEY = gtsam.symbol("C", 0)

        # 노이즈 모델
        # 오도메트리/GPS 측정 노이즈 (가정)
        pose_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.01)

        # 루프: 각 시점(t)마다 제약 조건 추가
        # T_w_c = T_w_b * T_b_c
        # => T_w_b.inverse() * T_w_c = T_b_c
        # 즉, BetweenFactor를 사용하여 (Body Pose)와 (Camera Pose) 사이의 관계가 (Extrinsics)가 되도록 함

        # 주의: GTSAM 그래프에서 Pose Key를 따로 만들지 않고,
        # 측정값(body_pose, cam_pose)을 상수로 취급하여 Prior처럼 넣는 방식이 가장 간단함.
        # 하지만 BetweenFactor는 두 변수 사이를 연결하는 것이므로,
        # 여기서는 T_bc라는 하나의 변수를 여러 "측정된 관계"들이 당기는 형태로 구현.

        # 변형된 방식:
        # 매 시간 t마다: Measured(T_b_c_meas) = (T_w_b)^-1 * (T_w_c)
        # 이 측정값들을 모두 만족하는 하나의 T_bc를 구함.

        for i in range(len(body_poses)):
            T_wb = body_poses[i]
            T_wc = camera_poses[i]

            # 측정된 상대 포즈 (Body -> Camera)
            # T_bc_measured = inv(T_wb) * T_wc
            T_bc_meas = T_wb.inverse().compose(T_wc)

            # PriorFactor를 사용하여 "이 시점에서 계산한 T_bc는 이거여야 해"라고 제약을 걺
            # 여러 시간대의 데이터가 쌓이면서 평균적인 최적해를 찾게 됨
            graph.add(gtsam.PriorFactorPose3(CALIB_KEY, T_bc_meas, pose_noise))

        # 초기값 삽입
        initial_estimates.insert(CALIB_KEY, initial_guess_bc)

        # 최적화
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates)
        result = optimizer.optimize()

        return result.atPose3(CALIB_KEY)
