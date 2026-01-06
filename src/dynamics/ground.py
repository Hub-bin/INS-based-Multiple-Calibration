import gtsam


class GroundVehicle:
    """
    지상 이동체(Ground Vehicle)의 운동학적 모델을 시뮬레이션하는 클래스입니다.
    GTSAM의 Pose3를 사용하여 3차원 공간상의 위치와 자세를 추적합니다.
    """

    def __init__(self, start_pose: gtsam.Pose3 = gtsam.Pose3()):
        """
        초기화 함수
            :param start_pose: 차량의 초기 위치 및 자세 (기본값: 원점)
        """
        self.current_pose = start_pose
        self.time = 0.0

        # 궤적 저장용 리스트 (Ground Truth)
        self.timestamps = [self.time]
        self.poses = [self.current_pose]

    def update(self, dt: float, velocity: float, yaw_rate: float):
        """
        시간 스텝(dt)만큼 상태를 업데이트합니다. (Body Frame 기준 이동)

        :param dt: 시간 간격 (seconds)
        :param velocity: 전진 속도 (m/s), Body Frame x-axis
        :param yaw_rate: 회전 각속도 (rad/s), Body Frame z-axis
        """
        # 1. Body Frame 기준에서의 미소 변위 생성
        # 지상 이동체이므로 Pitch, Roll 변화는 0으로 가정, Yaw만 변화
        # GTSAM Rot3.Ypr(yaw, pitch, roll) 순서 주의
        delta_rot = gtsam.Rot3.Ypr(yaw_rate * dt, 0.0, 0.0)
        delta_point = gtsam.Point3(velocity * dt, 0.0, 0.0)

        delta_pose = gtsam.Pose3(delta_rot, delta_point)

        # 2. 현재 Pose에 미소 변위를 합성 (Compose)
        # T_world_new = T_world_current * T_body_delta
        self.current_pose = self.current_pose.compose(delta_pose)
        self.time += dt

        # 3. 이력 저장
        self.timestamps.append(self.time)
        self.poses.append(self.current_pose)

    def get_trajectory(self):
        """저장된 전체 궤적 데이터를 반환합니다."""
        return self.timestamps, self.poses
