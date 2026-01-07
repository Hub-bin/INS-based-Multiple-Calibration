import gtsam
import numpy as np


class AerialVehicle:
    """
    공중 이동체(Aerial Vehicle)의 운동학적 모델 (예: 드론)
    - 3차원 공간을 자유롭게 이동하며, Roll/Pitch/Yaw 회전이 모두 발생합니다.
    """

    def __init__(self, start_pose: gtsam.Pose3 = gtsam.Pose3()):
        self.current_pose = start_pose
        self.time = 0.0

        # 궤적 저장
        self.timestamps = [self.time]
        self.poses = [self.current_pose]

    def update(self, dt: float, velocity_body: np.ndarray, omega_body: np.ndarray):
        """
        :param dt: 시간 간격
        :param velocity_body: Body Frame 기준 속도 [vx, vy, vz]
        :param omega_body: Body Frame 기준 각속도 [wx, wy, wz]
        """
        # 1. 회전 변화 (Body Frame 각속도 -> Rotation)
        # GTSAM Rot3.Expmap은 축-각(Axis-Angle) 변환을 수행
        delta_rot = gtsam.Rot3.Expmap(omega_body * dt)

        # 2. 위치 변화 (Body Frame 속도 -> Point3)
        delta_point = gtsam.Point3(
            velocity_body[0] * dt, velocity_body[1] * dt, velocity_body[2] * dt
        )

        # 3. Pose 합성 (T_new = T_curr * T_delta)
        delta_pose = gtsam.Pose3(delta_rot, delta_point)
        self.current_pose = self.current_pose.compose(delta_pose)
        self.time += dt

        # 저장
        self.timestamps.append(self.time)
        self.poses.append(self.current_pose)

    def get_trajectory(self):
        return self.timestamps, self.poses
