import numpy as np
from scipy.optimize import minimize


class SysIdCalibrator:
    def __init__(self, ref_temp=20.0):
        self.ref_temp = ref_temp

    def calibrate_sensor(self, true_data, raw_data, temp_data, active_mask=None):
        full_dim = 21
        if active_mask is None:
            active_indices = np.arange(full_dim)
        else:
            if len(active_mask) < full_dim:
                active_mask = np.concatenate([active_mask, np.zeros(full_dim - len(active_mask))])
            active_indices = np.where(np.array(active_mask) > 0.5)[0]

        if len(active_indices) == 0:
            return np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)

        x0_full = np.concatenate([np.eye(3).flatten(), np.zeros(12)])
        x0_reduced = x0_full[active_indices]

        dt_vec = (temp_data - self.ref_temp)[:, np.newaxis]
        dt2_vec = dt_vec**2

        # [핵심] ImuSensor와 동일한 Feature 생성
        # True Data의 변화량 계산
        true_diff = np.diff(true_data, axis=0, prepend=true_data[0:1])

        # ImuSensor와 동일하게 tanh 적용 (Scaling factor 100.0도 일치시킴)
        hyst_feature = np.tanh(true_diff * 100.0)

        def loss_func(x_reduced):
            x_curr = x0_full.copy()
            x_curr[active_indices] = x_reduced

            T = x_curr[:9].reshape(3, 3)
            b = x_curr[9:12]
            k_lin = x_curr[12:15]
            k_non = x_curr[15:18]
            k_hyst = x_curr[18:21]

            # Predict: Meas = T*True + Bias + ... + Hyst * Feature
            pred = (
                (true_data @ T.T)
                + b
                + (k_lin * dt_vec)
                + (k_non * dt2_vec)
                + (k_hyst * hyst_feature)
            )

            return np.sum((raw_data - pred) ** 2)

        res = minimize(loss_func, x0_reduced, method="L-BFGS-B")

        x_final = x0_full.copy()
        if res.success:
            x_final[active_indices] = res.x

        return (
            x_final[:9].reshape(3, 3),
            x_final[9:12],
            x_final[12:15],
            x_final[15:18],
            x_final[18:21],
        )

    def run(
        self, true_measurements, raw_measurements, temp_measurements, acc_mask=None, gyr_mask=None
    ):
        def parse(data):
            if isinstance(data, list):
                return np.array([d[0] for d in data]), np.array([d[1] for d in data])
            return data[:, :3], data[:, 3:]

        t_acc, t_gyr = parse(true_measurements)
        r_acc, r_gyr = parse(raw_measurements)
        temps = np.array(temp_measurements)

        ar = self.calibrate_sensor(t_acc, r_acc, temps, acc_mask)
        gr = self.calibrate_sensor(t_gyr, r_gyr, temps, gyr_mask)

        return {
            "acc_T_inv": ar[0],
            "acc_b": ar[1],
            "acc_k1": ar[2],
            "acc_k2": ar[3],
            "acc_h": ar[4],
            "gyr_T_inv": gr[0],
            "gyr_b": gr[1],
            "gyr_k1": gr[2],
            "gyr_k2": gr[3],
            "gyr_h": gr[4],
        }
