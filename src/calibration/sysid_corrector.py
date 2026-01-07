import numpy as np
from scipy.optimize import minimize


class SysIdCalibrator:
    """
    수치 최적화(Optimization) 기반의 시스템 식별 교정기
    - [Final Update] Hysteresis (History dependent error) 추정 기능 추가
    - Parameter Vector (Size 27):
      [0~8: Matrix(9), 9~11: Bias(3), 12~14: TempLin(3), 15~17: TempNon(3), 18~20: Hysteresis(3)]
    """

    def __init__(self, ref_temp=20.0):
        self.ref_temp = ref_temp

    def calibrate_sensor(self, true_data, raw_data, temp_data, active_mask=None):
        """
        :param active_mask: (21,) Mask including Hysteresis
        """
        # 파라미터 개수: 9(Mat) + 3(Bias) + 3(T_Lin) + 3(T_Non) + 3(Hyst) = 21 per sensor
        full_dim = 21

        if active_mask is None:
            active_indices = np.arange(full_dim)
        else:
            # Mask 사이즈가 안맞으면 패딩 (이전 코드 호환성)
            if len(active_mask) < full_dim:
                padding = np.zeros(full_dim - len(active_mask))
                active_mask = np.concatenate([active_mask, padding])
            active_indices = np.where(np.array(active_mask) > 0.5)[0]

        if len(active_indices) == 0:
            return np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)

        # 초기값
        x0_full = np.concatenate(
            [
                np.eye(3).flatten(),  # 0~8
                np.zeros(3),  # 9~11 (Bias)
                np.zeros(3),  # 12~14 (TempLin)
                np.zeros(3),  # 15~17 (TempNon)
                np.zeros(3),  # 18~20 (Hysteresis)
            ]
        )

        x0_reduced = x0_full[active_indices]

        # Pre-compute features
        delta_t = temp_data - self.ref_temp
        delta_t_sq = delta_t**2
        dt_vec = delta_t[:, np.newaxis]
        dt2_vec = delta_t_sq[:, np.newaxis]

        # [New] Hysteresis Feature: sign(Delta Raw)
        # Raw 데이터의 변화 방향을 미리 계산 (가속 중인가? 감속 중인가?)
        # [0, d1, d2, ...]
        raw_diff = np.diff(raw_data, axis=0, prepend=raw_data[0:1])
        # 노이즈에 민감하지 않게 Thresholding
        hyst_sign = np.sign(raw_diff)
        # (N, 3)

        def loss_func(x_reduced):
            x_curr = x0_full.copy()
            x_curr[active_indices] = x_reduced

            # Unpack
            T = x_curr[:9].reshape(3, 3)
            b = x_curr[9:12]
            k_lin = x_curr[12:15]
            k_non = x_curr[15:18]
            k_hyst = x_curr[18:21]  # Hysteresis Magnitude

            # Predict:
            # Meas ~= T*True + Bias + TempEff + HystEff
            # HystEff = k_hyst * sign(change)

            pred = (
                (true_data @ T.T) + b + (k_lin * dt_vec) + (k_non * dt2_vec) + (k_hyst * hyst_sign)
            )

            diff = raw_data - pred
            return np.sum(diff**2)

        # Optimization
        result = minimize(loss_func, x0_reduced, method="L-BFGS-B")

        x_final = x0_full.copy()
        if result.success:
            x_final[active_indices] = result.x

            opt_T = x_final[:9].reshape(3, 3)
            opt_b = x_final[9:12]
            opt_klin = x_final[12:15]
            opt_knon = x_final[15:18]
            opt_khyst = x_final[18:21]

            try:
                T_inv = np.linalg.inv(opt_T)
            except:
                T_inv = np.eye(3)

            return T_inv, opt_b, opt_klin, opt_knon, opt_khyst
        else:
            return np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)

    def run(
        self, true_measurements, raw_measurements, temp_measurements, acc_mask=None, gyr_mask=None
    ):
        true_acc = np.array([m[0] for m in true_measurements])
        true_gyr = np.array([m[1] for m in true_measurements])
        raw_acc = np.array([m[0] for m in raw_measurements])
        raw_gyr = np.array([m[1] for m in raw_measurements])
        temps = np.array(temp_measurements)

        acc_res = self.calibrate_sensor(true_acc, raw_acc, temps, acc_mask)
        gyr_res = self.calibrate_sensor(true_gyr, raw_gyr, temps, gyr_mask)

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
