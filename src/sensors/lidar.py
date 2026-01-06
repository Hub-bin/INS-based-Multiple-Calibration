import gtsam
import numpy as np


class LidarSensor:
    """
    3D LiDAR 센서 시뮬레이터
    - 주변 랜드마크 중 센서 범위(Max Range)와 수직 시야각(Vertical FOV) 내에 있는 점들을 감지합니다.
    - 출력은 LiDAR 좌표계 기준의 3D Point Cloud입니다.
    """

    def __init__(self, max_range=50.0, v_fov=30.0, range_noise=0.05, angle_noise=0.01):
        """
        :param max_range: 최대 감지 거리 (m)
        :param v_fov: 수직 화각 (Degree, 예: 30도면 위아래 ±15도)
        :param range_noise: 거리 측정 노이즈 표준편차 (m)
        :param angle_noise: 각도 측정 노이즈 표준편차 (rad)
        """
        self.max_range = max_range
        self.min_vertical_angle = np.radians(-v_fov / 2)
        self.max_vertical_angle = np.radians(v_fov / 2)

        self.range_noise_sigma = range_noise
        self.angle_noise_sigma = angle_noise

        # LiDAR 설치 위치 (차량 중심에서 위로 1.5m)
        self.body_to_lidar = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0.0, 0.0, 1.5))

    def measure(self, vehicle_pose_world, landmarks):
        """
        :param vehicle_pose_world: 차량의 현재 Pose
        :param landmarks: 전체 랜드마크 딕셔너리
        :return: measurements (LiDAR 좌표계 기준의 3D Point List)
        """
        scanned_points = []

        # World Frame -> Lidar Frame 변환 행렬
        lidar_pose_world = vehicle_pose_world.compose(self.body_to_lidar)
        world_to_lidar = lidar_pose_world.inverse()

        for point_world in landmarks.values():
            # 1. 좌표 변환 (World -> LiDAR Local)
            # [수정됨] point_local은 이제 gtsam.Point3가 아니라 numpy array([x, y, z])입니다.
            point_local = world_to_lidar.transformFrom(point_world)

            # 2. 거리 체크 (Max Range)
            # numpy 배열이므로 np.linalg.norm을 바로 사용 가능
            distance = np.linalg.norm(point_local)

            if distance > self.max_range:
                continue

            # 3. 각도 체크 (Vertical FOV)
            # 배열 인덱싱 사용: z -> point_local[2]
            elevation = np.arcsin(point_local[2] / distance)

            if self.min_vertical_angle <= elevation <= self.max_vertical_angle:
                # 4. 노이즈 추가 (극좌표계)
                # 배열 인덱싱 사용: x -> [0], y -> [1]
                azimuth = np.arctan2(point_local[1], point_local[0])

                noisy_dist = distance + np.random.normal(0, self.range_noise_sigma)
                noisy_azimuth = azimuth + np.random.normal(0, self.angle_noise_sigma)
                noisy_elevation = elevation + np.random.normal(0, self.angle_noise_sigma)

                # 다시 3D 좌표로 복원 (Polar -> Cartesian)
                lx = noisy_dist * np.cos(noisy_elevation) * np.cos(noisy_azimuth)
                ly = noisy_dist * np.cos(noisy_elevation) * np.sin(noisy_azimuth)
                lz = noisy_dist * np.sin(noisy_elevation)

                scanned_points.append(gtsam.Point3(lx, ly, lz))

        return scanned_points, lidar_pose_world
