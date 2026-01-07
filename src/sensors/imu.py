import numpy as np
import gtsam


class ImuSensor:
    def __init__(
        self,
        accel_bias=None,
        accel_noise=0.0,
        gyro_noise=0.0,
        accel_temp_coeff_linear=0.0,
        accel_temp_coeff_nonlinear=0.0,
        accel_hysteresis=0.0,
        gyro_bias=None,
    ):
        self.accel_bias = np.array(accel_bias) if accel_bias is not None else np.zeros(3)
        self.gyro_bias = np.array(gyro_bias) if gyro_bias is not None else np.zeros(3)
        self.accel_noise_std = accel_noise
        self.gyro_noise_std = gyro_noise

        # Temperature & Hysteresis Params
        self.acc_k1 = accel_temp_coeff_linear
        self.acc_k2 = accel_temp_coeff_nonlinear
        self.acc_h = accel_hysteresis

        # State for Hysteresis (Previous true acceleration)
        self.prev_accel = None

    def measure(self, pose, accel_body, omega_body, temperature=20.0):
        # 1. True Values
        true_acc = accel_body
        true_gyr = omega_body

        # 2. Hysteresis Effect
        # Logic: H * sign(Current - Previous)
        # 가속도가 증가하면 +H, 감소하면 -H 오차 추가 (Lagging Loop)
        if self.prev_accel is None:
            hyst_error = np.zeros(3)
        else:
            diff = true_acc - self.prev_accel
            # tanh를 사용하여 부드러운 스위칭 (최적화에 유리)
            # diff가 0.01 이상이면 sign이 1에 가까워짐
            hyst_error = self.acc_h * np.tanh(diff * 100.0)

        self.prev_accel = true_acc  # Update state

        # 3. Temperature Effect
        dt = temperature - 20.0
        temp_error = (self.acc_k1 * dt) + (self.acc_k2 * (dt**2))

        # 4. Total Error Model
        # Meas = True + Bias + Temp + Hyst + Noise
        noise_acc = np.random.normal(0, self.accel_noise_std, 3)
        noise_gyr = np.random.normal(0, self.gyro_noise_std, 3)

        meas_acc = true_acc + self.accel_bias + temp_error + hyst_error + noise_acc
        meas_gyr = true_gyr + self.gyro_bias + noise_gyr

        return meas_acc, meas_gyr, {}
