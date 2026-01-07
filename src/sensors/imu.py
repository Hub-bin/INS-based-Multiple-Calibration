import gtsam
import numpy as np


class ImuSensor:
    """
    6-DOF IMU 센서 시뮬레이터 (Enhanced)
    - Bias, Scale Factor, Misalignment, Noise를 모두 시뮬레이션
    - 수식: measured = (I + S + M) * true + bias + noise
    """

    def __init__(
        self,
        accel_noise=0.01,
        gyro_noise=0.001,
        accel_bias=None,
        gyro_bias=None,
        accel_error_matrix=None,
        gyro_error_matrix=None,
    ):
        """
        :param accel_error_matrix: 가속도계 Scale+Misalignment 행렬 (3x3)
        :param gyro_error_matrix: 자이로 Scale+Misalignment 행렬 (3x3)
        """
        self.accel_noise_sigma = accel_noise
        self.gyro_noise_sigma = gyro_noise

        # Bias (Offset)
        self.a_bias = np.array(accel_bias) if accel_bias is not None else np.zeros(3)
        self.g_bias = np.array(gyro_bias) if gyro_bias is not None else np.zeros(3)

        # Scale Factor + Misalignment Matrix (T)
        # 값이 없으면 단위 행렬(Identity)로 설정
        if accel_error_matrix is None:
            self.T_accel = np.eye(3)
        else:
            self.T_accel = np.array(accel_error_matrix)

        if gyro_error_matrix is None:
            self.T_gyro = np.eye(3)
        else:
            self.T_gyro = np.array(gyro_error_matrix)

        # GTSAM Bias 객체 (Noise 없는 순수 Bias만 관리)
        self.bias = gtsam.imuBias.ConstantBias(self.a_bias, self.g_bias)

        # 중력 가속도
        self.gravity_world = np.array([0, 0, -9.81])

    def measure(self, pose: gtsam.Pose3, true_accel_body: np.ndarray, true_omega_body: np.ndarray):
        # 1. True Specific Force 계산
        rot_wb = pose.rotation()
        # [수정] numpy array 반환 대응
        gravity_body = rot_wb.unrotate(gtsam.Point3(*self.gravity_world))

        # Point3 객체일 경우 numpy 변환
        if not isinstance(gravity_body, np.ndarray):
            gravity_body = np.array([gravity_body.x(), gravity_body.y(), gravity_body.z()])

        true_specific_force = true_accel_body - gravity_body

        # 2. Scale Factor & Misalignment 적용 (행렬 연산)
        # meas = T * true
        distorted_accel = self.T_accel @ true_specific_force
        distorted_omega = self.T_gyro @ true_omega_body

        # 3. Bias 추가
        distorted_accel += self.a_bias
        distorted_omega += self.g_bias

        # 4. Noise 추가
        noisy_accel = distorted_accel + np.random.normal(0, self.accel_noise_sigma, 3)
        noisy_omega = distorted_omega + np.random.normal(0, self.gyro_noise_sigma, 3)

        return noisy_accel, noisy_omega
