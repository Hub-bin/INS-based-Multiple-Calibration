import gtsam
import numpy as np


class LidarCameraCalibrator:
    """
    LiDAR-Camera Extrinsic Calibration 모듈
    - 3D-2D 매칭 쌍(LiDAR Point - Camera Pixel)을 사용하여
    - 두 센서 간의 변환 행렬(Extrinsics)을 최적화합니다.
    """

    def __init__(self, camera_intrinsics: gtsam.Cal3_S2):
        """
        :param camera_intrinsics: 카메라 내부 파라미터 (fx, fy, s, u0, v0)
        """
        self.intrinsics = camera_intrinsics

    def run(self, correspondences, initial_guess_pose: gtsam.Pose3):
        """
        최적화를 수행하여 보정된 Extrinsics를 반환합니다.

        :param correspondences: List of (point3_lidar, point2_pixel) tuples
        :param initial_guess_pose: 초기 추정 Extrinsics (T_cam_lidar)
                                   (LiDAR 좌표계 -> Camera 좌표계 변환 행렬)
        :return: optimized_pose (gtsam.Pose3)
        """
        graph = gtsam.NonlinearFactorGraph()
        initial_estimates = gtsam.Values()

        # 최적화 변수 Key (Extrinsic Pose)
        POSE_KEY = gtsam.symbol("x", 0)

        # 1. Noise Model 설정 (픽셀 측정 노이즈)
        # 픽셀 단위이므로 보통 1.0 ~ 2.0 픽셀 정도의 오차를 가정
        pixel_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)

        # 랜드마크를 고정시키기 위한 강력한 노이즈 모델 (Constrained)
        # 0에 가까운 아주 작은 분산을 의미 -> 최적화 시 값이 변하지 않음
        point_fixed_noise = gtsam.noiseModel.Constrained.All(3)

        # 2. Factor 및 변수 추가
        for i, (point_lidar, point_pixel) in enumerate(correspondences):
            # 2-1. 랜드마크 Key 생성 (l0, l1, l2 ...)
            point_key = gtsam.symbol("l", i)

            # 2-2. Numpy array -> gtsam.Point3 변환
            if isinstance(point_lidar, np.ndarray):
                pt = gtsam.Point3(point_lidar[0], point_lidar[1], point_lidar[2])
            else:
                pt = point_lidar

            # 2-3. GenericProjectionFactor 추가 (Pose Key와 Point Key를 연결)
            factor = gtsam.GenericProjectionFactorCal3_S2(
                point_pixel,  # 관측된 2D 픽셀
                pixel_noise,  # 픽셀 노이즈
                POSE_KEY,  # 최적화할 Pose 변수 Key
                point_key,  # 랜드마크 변수 Key (수정됨: 객체 대신 Key 전달)
                self.intrinsics,  # 내부 파라미터
            )
            graph.add(factor)

            # 2-4. 랜드마크 고정 (Fix Landmarks)
            # 우리는 Extrinsics만 구하고 싶으므로 랜드마크 위치는 Ground Truth로 고정합니다.
            # 이를 위해 PriorFactor를 추가하고 Noise를 Constrained(0)로 설정합니다.
            graph.add(gtsam.PriorFactorPoint3(point_key, pt, point_fixed_noise))

            # 2-5. 랜드마크 초기값 추가
            initial_estimates.insert(point_key, pt)

        # 3. Pose Prior Factor (Weak Prior)
        # 초기값이 너무 튀는 것을 방지하기 위한 아주 약한 Prior
        pose_prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, 100.0)
        graph.add(gtsam.PriorFactorPose3(POSE_KEY, initial_guess_pose, pose_prior_noise))

        # 4. Pose 초기값 삽입
        initial_estimates.insert(POSE_KEY, initial_guess_pose)

        # 5. 최적화 실행 (Levenberg-Marquardt)
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates)
        result = optimizer.optimize()

        return result.atPose3(POSE_KEY)
