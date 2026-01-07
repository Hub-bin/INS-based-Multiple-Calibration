import gtsam
import numpy as np


class ImuSensor:
    """
    6-DOF IMU 센서 시뮬레이터 (Final Enhanced)
    - Bias, Scale Factor, Misalignment, Noise
    - Temperature Dependent Bias Drift
    - [New] Hysteresis Effect (History Dependent Error)
    """

    def __init__(
        self,
        accel_noise=0.01,
        gyro_noise=0.001,
        accel_bias=None,
        gyro_bias=None,
        accel_error_matrix=None,
        gyro_error_matrix=None,
        # 온도 파라미터
        ref_temperature=20.0,
        accel_temp_coeff_linear=0.0,
        accel_temp_coeff_nonlinear=0.0,
        gyro_temp_coeff_linear=0.0,
        gyro_temp_coeff_nonlinear=0.0,
        # [New] 히스테리시스 파라미터
        accel_hysteresis=0.0,  # (m/s^2) Max hysteresis deviation
    ):
        self.accel_noise_sigma = accel_noise
        self.gyro_noise_sigma = gyro_noise

        # Base Bias
        self.a_bias_base = np.array(accel_bias) if accel_bias is not None else np.zeros(3)
        self.g_bias_base = np.array(gyro_bias) if gyro_bias is not None else np.zeros(3)

        # Scale & Misalignment
        self.T_accel = np.array(accel_error_matrix) if accel_error_matrix is not None else np.eye(3)
        self.T_gyro = np.array(gyro_error_matrix) if gyro_error_matrix is not None else np.eye(3)

        # Temperature
        self.ref_temp = ref_temperature
        self.acc_tc_lin = (
            np.array(accel_temp_coeff_linear)
            if np.ndim(accel_temp_coeff_linear) > 0
            else np.full(3, accel_temp_coeff_linear)
        )
        self.acc_tc_non = (
            np.array(accel_temp_coeff_nonlinear)
            if np.ndim(accel_temp_coeff_nonlinear) > 0
            else np.full(3, accel_temp_coeff_nonlinear)
        )

        self.gyr_tc_lin = (
            np.array(gyro_temp_coeff_linear)
            if np.ndim(gyro_temp_coeff_linear) > 0
            else np.full(3, gyro_temp_coeff_linear)
        )
        self.gyr_tc_non = (
            np.array(gyro_temp_coeff_nonlinear)
            if np.ndim(gyro_temp_coeff_nonlinear) > 0
            else np.full(3, gyro_temp_coeff_nonlinear)
        )

        # Hysteresis State
        self.acc_hyst_mag = accel_hysteresis
        self.prev_true_accel = None
        # 현재의 자기적/기계적 상태 (Memory) - 간단히 누적된 Lag 값으로 모델링
        self.hysteresis_state = np.zeros(3)

        # Gravity
        self.gravity_world = np.array([0, 0, -9.81])

    def _compute_temp_bias(self, current_temp):
        delta_t = current_temp - self.ref_temp
        drift_acc = (self.acc_tc_lin * delta_t) + (self.acc_tc_non * (delta_t**2))
        drift_gyr = (self.gyr_tc_lin * delta_t) + (self.gyr_tc_non * (delta_t**2))
        return drift_acc, drift_gyr

    def _compute_hysteresis(self, current_true_accel):
        """
        가속도 변화 방향에 따라 Hysteresis 오차 계산 (Simplified Preisach Model)
        - 가속도가 증가하면 출력은 아래쪽 경로, 감소하면 위쪽 경로를 따름 (Lagging)
        """
        if self.acc_hyst_mag == 0.0:
            return np.zeros(3)

        if self.prev_true_accel is None:
            self.prev_true_accel = current_true_accel
            return np.zeros(3)

        # 변화량
        diff = current_true_accel - self.prev_true_accel

        # 방향에 따라 상태 업데이트 (Lag effect)
        # alpha: 상태가 입력을 따라가는 속도 (0~1)
        alpha = 0.1
        target_hyst = -np.sign(diff) * self.acc_hyst_mag

        # diff가 0에 가까우면(정지) 상태 유지
        # 움직임이 있을 때만 Hysteresis Loop를 그림
        mask = (np.abs(diff) > 1e-5).astype(float)

        self.hysteresis_state = self.hysteresis_state * (1 - mask * alpha) + target_hyst * (
            mask * alpha
        )

        self.prev_true_accel = current_true_accel
        return self.hysteresis_state

    def measure(
        self,
        pose: gtsam.Pose3,
        true_accel_body: np.ndarray,
        true_omega_body: np.ndarray,
        temperature=20.0,
    ):
        # 1. True Specific Force
        rot_wb = pose.rotation()
        g_world = gtsam.Point3(*self.gravity_world)
        g_body_gtsam = rot_wb.unrotate(g_world)

        if isinstance(g_body_gtsam, np.ndarray):
            g_body = g_body_gtsam
        else:
            g_body = np.array([g_body_gtsam.x(), g_body_gtsam.y(), g_body_gtsam.z()])

        true_specific_force = true_accel_body - g_body

        # 2. Scale & Misalignment
        distorted_accel = self.T_accel @ true_specific_force
        distorted_omega = self.T_gyro @ true_omega_body

        # 3. Bias + Temperature Drift
        temp_drift_acc, temp_drift_gyr = self._compute_temp_bias(temperature)

        # 4. [New] Hysteresis Effect
        hyst_error = self._compute_hysteresis(true_specific_force)

        final_acc_bias = self.a_bias_base + temp_drift_acc + hyst_error
        final_gyr_bias = self.g_bias_base + temp_drift_gyr

        distorted_accel += final_acc_bias
        distorted_omega += final_gyr_bias

        # 5. Noise
        noisy_accel = distorted_accel + np.random.normal(0, self.accel_noise_sigma, 3)
        noisy_omega = distorted_omega + np.random.normal(0, self.gyro_noise_sigma, 3)

        return noisy_accel, noisy_omega, (final_acc_bias, final_gyr_bias)
