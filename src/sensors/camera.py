import gtsam
import numpy as np


class CameraSensor:
    """
    Pinhole Camera 모델 시뮬레이터
    - 3D 랜드마크를 2D 이미지 평면으로 투영(Project)하여 관측 데이터를 생성합니다.
    """

    def __init__(self, noise_sigma=1.0, width=640, height=480, fov=90.0):
        """
        :param noise_sigma: 픽셀 측정 노이즈 표준편차 (pixels)
        :param width: 이미지 가로 해상도
        :param height: 이미지 세로 해상도
        :param fov: 수평 화각 (Degree)
        """
        self.noise_sigma = noise_sigma
        self.width = width
        self.height = height

        # 내부 파라미터 (Intrinsics) 계산
        # fx = width / (2 * tan(fov / 2))
        fx = width / (2 * np.tan(np.radians(fov) / 2))
        fy = fx  # Square pixels 가정
        u0 = width / 2
        v0 = height / 2

        # GTSAM Cal3_S2 (fx, fy, s, u0, v0) - 왜곡 없음 가정
        self.calibration = gtsam.Cal3_S2(fx, fy, 0.0, u0, v0)

        # 카메라와 차량(Body) 사이의 변환 행렬 (Extrinsics)
        # 예: 차량 중심에서 앞쪽으로 2m, 위로 1m, 카메라는 앞을 정면으로 바라봄(X-forward)
        # GTSAM의 카메라는 보통 Z-forward (OpenCV 스타일)이므로 좌표계 변환 필요
        # Body(X-forward, Z-up) -> Camera(Z-forward, Y-down)
        # R_bc = [ [0, -1, 0], [0, 0, -1], [1, 0, 0] ]
        R_bc = gtsam.Rot3(np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]))
        t_bc = gtsam.Point3(2.0, 0.0, 1.0)  # 차량 중심에서 전방 2m, 상방 1m 설치
        self.body_to_camera = gtsam.Pose3(R_bc, t_bc)

    def measure(self, vehicle_pose_world, landmarks):
        """
        현재 차량 위치에서 보이는 랜드마크들을 관측합니다.

        :param vehicle_pose_world: 차량의 현재 Pose (World Frame)
        :param landmarks: 랜드마크 딕셔너리 {id: Point3(x, y, z)}
        :return: measurements list [(landmark_id, Point2(u, v)), ...]
        """
        measurements = []

        # World Frame의 카메라 Pose 계산
        # T_wc = T_wb * T_bc
        camera_pose_world = vehicle_pose_world.compose(self.body_to_camera)

        # GTSAM PinholeCamera 객체 생성
        camera = gtsam.PinholeCameraCal3_S2(camera_pose_world, self.calibration)

        for l_id, point_world in landmarks.items():
            try:
                # 1. Cheirality Check (카메라 뒤에 있는지 확인)
                # GTSAM은 점이 카메라 뒤에 있으면 에러를 던지거나 투영하지 않음
                # project() 함수 내부적으로 체크하지만, 여기서는 범위를 명확히 하기 위해 try-except 사용

                # 2. 투영 (Projection)
                uv = camera.project(point_world)

                # 3. FOV 체크 (이미지 해상도 안에 들어오는지)
                if (0 <= uv[0] < self.width) and (0 <= uv[1] < self.height):
                    # 4. 노이즈 추가
                    u_noise = np.random.normal(0, self.noise_sigma)
                    v_noise = np.random.normal(0, self.noise_sigma)
                    measured_uv = gtsam.Point2(uv[0] + u_noise, uv[1] + v_noise)

                    measurements.append((l_id, measured_uv))

            except RuntimeError:
                # 포인트가 카메라 뒤에 있음 (Cheirality Exception)
                pass

        return measurements, camera_pose_world
