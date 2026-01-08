import numpy as np
from scipy.optimize import minimize


class SysIdCalibrator:
    def __init__(self, ref_temp=20.0):
        self.ref_temp = ref_temp

    def calibrate_sensor(self, true_data, raw_data, temp_data, active_mask=None, is_accel=True):
        full_dim = 21

        if active_mask is None:
            active_indices = np.arange(full_dim)
        else:
            if len(active_mask) < full_dim:
                padding = np.zeros(full_dim - len(active_mask))
                active_mask = np.concatenate([active_mask, padding])
            active_indices = np.where(np.array(active_mask) > 0.5)[0]

        if len(active_indices) == 0:
            return np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)

        # 초기값: Scale은 1.0, 나머지는 0.0
        x0_full = np.zeros(full_dim)
        x0_full[0], x0_full[4], x0_full[8] = 1.0, 1.0, 1.0

        # [수정] Bias 초기값: 가속도계 Z축은 중력(9.81) 근처에서 시작하도록 유도
        if is_accel:
            x0_full[11] = 9.81

        x0_reduced = x0_full[active_indices]

        delta_t = temp_data - self.ref_temp
        delta_t_sq = delta_t**2
        dt_vec = delta_t[:, np.newaxis]
        dt2_vec = delta_t_sq[:, np.newaxis]

        true_diff = np.diff(true_data, axis=0, prepend=true_data[0:1])
        hyst_feature = np.tanh(true_diff * 10.0)

        def loss_func(x_reduced):
            x_curr = x0_full.copy()
            x_curr[active_indices] = x_reduced

            T = x_curr[:9].reshape(3, 3)
            b = x_curr[9:12]
            k_lin = x_curr[12:15]
            k_non = x_curr[15:18]
            k_hyst = x_curr[18:21]

            pred = (
                (true_data @ T.T)
                + b
                + (k_lin * dt_vec)
                + (k_non * dt2_vec)
                + (k_hyst * hyst_feature)
            )

            diff = raw_data - pred
            return np.sum(diff**2)

        # [수정] Bounds 추가: Scale은 0.5 ~ 1.5, 나머지는 자유
        # T matrix (0~8), Bias (9~11), ...
        bounds = []
        for i in range(full_dim):
            if i in [0, 4, 8]:  # Diagonals of T (Scale)
                bounds.append((0.5, 1.5))
            else:
                bounds.append((None, None))

        bounds_reduced = [bounds[i] for i in active_indices]

        # 최적화
        result = minimize(
            loss_func,
            x0_reduced,
            method="L-BFGS-B",
            bounds=bounds_reduced,
            options={"maxiter": 1000},
        )

        x_final = x0_full.copy()
        if result.success:
            x_final[active_indices] = result.x
        else:
            # 실패 시 경고 출력 후 최선값 사용
            # print(f"[SysID] Warning: Optimization failed ({result.message})")
            x_final[active_indices] = result.x

        opt_T = x_final[:9].reshape(3, 3)
        opt_b = x_final[9:12]
        # ... (나머지 동일) ...

        try:
            T_inv = np.linalg.inv(opt_T)
        except:
            T_inv = np.eye(3)

        return T_inv, opt_b, x_final[12:15], x_final[15:18], x_final[18:21]

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

        # [수정] is_accel 플래그 전달
        acc_res = self.calibrate_sensor(t_acc, r_acc, temps, acc_mask, is_accel=True)
        gyr_res = self.calibrate_sensor(t_gyr, r_gyr, temps, gyr_mask, is_accel=False)

        return {
            "acc_T_inv": acc_res[0],
            "acc_b": acc_res[1],
            "acc_k1": acc_res[2],
            "acc_k2": acc_res[3],
            "acc_h": acc_res[4],
            "gyr_T_inv": gyr_res[0],
            "gyr_b": gyr_res[1],
            "gyr_k1": gyr_res[2],
            "gyr_k2": gyr_res[3],
            "gyr_h": gyr_res[4],
        }
