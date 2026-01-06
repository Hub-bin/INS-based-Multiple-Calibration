import gtsam
import numpy as np


class ImuSensor:
    """
    6-DOF IMU 센서 시뮬레이터
    - Accelerometer: Body Frame 기준의 비력(Specific Force) 측정 (중력 포함)
    - Gyroscope: Body Frame 기준의 각속도 측정
    """

    def __init__(self, accel_noise=0.01, gyro_noise=0.001, accel_bias=None, gyro_bias=None):
        """
        :param accel_noise: 가속도계 화이트 노이즈 표준편차 (m/s^2)
        :param gyro_noise: 자이로스코프 화이트 노이즈 표준편차 (rad/s)
        :param accel_bias: 초기 가속도 바이어스 (x, y, z)
        :param gyro_bias: 초기 자이로 바이어스 (x, y, z)
        """
        self.accel_noise_sigma = accel_noise
        self.gyro_noise_sigma = gyro_noise

        # 초기 바이어스 설정 (없으면 0)
        a_bias = np.array(accel_bias) if accel_bias is not None else np.zeros(3)
        g_bias = np.array(gyro_bias) if gyro_bias is not None else np.zeros(3)

        # GTSAM의 ConstantBias 클래스 사용하여 관리 (최적화 시 편리)
        self.bias = gtsam.imuBias.ConstantBias(a_bias, g_bias)

        # 중력 가속도 (World Frame, Z-up)
        self.gravity_world = np.array([0, 0, -9.81])

    def measure(self, pose: gtsam.Pose3, true_accel_body: np.ndarray, true_omega_body: np.ndarray):
        """
        현재 상태를 기반으로 노이즈가 섞인 IMU 측정값을 생성합니다.

        :param pose: 현재 차량의 자세 (World Frame)
        :param true_accel_body: 차량의 실제 가속도 (Body Frame, 운동학적 가속도)
        :param true_omega_body: 차량의 실제 각속도 (Body Frame)
        :return: (measured_accel, measured_omega)
        """
        # 1. 가속도계 모델링 (Specific Force = Kinematic Accel - Gravity)
        # a_meas = R_wb^T * (a_world - g_world) + bias + noise
        # 여기서 a_world를 Body Frame으로 변환한 것이 true_accel_body라고 가정하면 로직이 복잡해지므로,
        # 수식: a_meas_body = true_accel_body - R_wb^T * g_world

        # World Frame의 중력 벡터를 Body Frame으로 회전 (Unrotate)
        rot_wb = pose.rotation()
        gravity_body = rot_wb.unrotate(gtsam.Point3(*self.gravity_world))

        # 비력(Specific Force) 계산: 중력이 아래로 당기면 센서는 위로 가속한다고 느낌 (+9.81 on Z when flat)
        true_specific_force = true_accel_body - gravity_body

        # 2. 바이어스 추가
        noisy_accel = true_specific_force + self.bias.accelerometer()
        noisy_omega = true_omega_body + self.bias.gyroscope()

        # 3. 화이트 노이즈 추가 (Gaussian)
        noisy_accel += np.random.normal(0, self.accel_noise_sigma, 3)
        noisy_omega += np.random.normal(0, self.gyro_noise_sigma, 3)

        return noisy_accel, noisy_omega
