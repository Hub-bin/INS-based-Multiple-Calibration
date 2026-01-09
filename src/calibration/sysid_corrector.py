import numpy as np
from scipy.optimize import minimize


class SysIdCalibrator:
    def __init__(self, ref_temp=20.0):
        self.ref_temp = ref_temp

    # [수정] initial_guess와 prev_true 인자 추가
    def calibrate_sensor(
        self, true_data, raw_data, temp_data, active_mask=None, initial_guess=None, prev_true=None
    ):
        full_dim = 21

        # 마스크 처리
        if active_mask is None:
            active_indices = np.arange(full_dim)
        else:
            if len(active_mask) < full_dim:
                padding = np.zeros(full_dim - len(active_mask))
                active_mask = np.concatenate([active_mask, padding])
            active_indices = np.where(np.array(active_mask) > 0.5)[0]

        if len(active_indices) == 0:
            return np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(full_dim)

        # [핵심] 초기값 설정 (Warm Start)
        if initial_guess is not None and len(initial_guess) == full_dim:
            x0_full = np.array(initial_guess, dtype=float)
        else:
            # Cold Start
            x0_full = np.zeros(full_dim)
            x0_full[0], x0_full[4], x0_full[8] = 1.0, 1.0, 1.0  # Scale 1.0

        x0_reduced = x0_full[active_indices]

        # 온도 변수
        delta_t = temp_data - self.ref_temp
        dt_vec = delta_t[:, np.newaxis]
        dt2_vec = (delta_t**2)[:, np.newaxis]

        # [핵심] 히스테리시스 피처 (이전 데이터 연결)
        if prev_true is not None:
            # 윈도우 직전 값과 현재 첫 값의 차이로 시작
            prepend_val = prev_true.reshape(1, 3)
            true_diff = np.diff(true_data, axis=0, prepend=prepend_val)
        else:
            # 정보 없으면 0으로 가정 (기존 방식)
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

        # Bounds
        bounds = []
        for i in range(full_dim):
            if i in [0, 4, 8]:
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

        # [수정] 다음 Warm Start를 위해 전체 파라미터 벡터(x_final)도 반환
        return T_inv, opt_b, opt_klin, opt_knon, opt_khyst, x_final

    # [수정] run 인터페이스 확장
    def run(
        self,
        true_measurements,
        raw_measurements,
        temp_measurements,
        acc_mask=None,
        gyr_mask=None,
        init_acc_params=None,
        init_gyr_params=None,
        prev_true_acc=None,
        prev_true_gyr=None,
    ):
        def parse(data):
            if isinstance(data, list):
                return np.array([d[0] for d in data]), np.array([d[1] for d in data])
            return data[:, :3], data[:, 3:]

        t_acc, t_gyr = parse(true_measurements)
        r_acc, r_gyr = parse(raw_measurements)
        temps = np.array(temp_measurements)

        # 각각 Warm Start 인자 전달
        acc_res = self.calibrate_sensor(
            t_acc, r_acc, temps, acc_mask, initial_guess=init_acc_params, prev_true=prev_true_acc
        )
        gyr_res = self.calibrate_sensor(
            t_gyr, r_gyr, temps, gyr_mask, initial_guess=init_gyr_params, prev_true=prev_true_gyr
        )

        return {
            "acc_T_inv": acc_res[0],
            "acc_b": acc_res[1],
            "acc_k1": acc_res[2],
            "acc_k2": acc_res[3],
            "acc_h": acc_res[4],
            "acc_params": acc_res[5],
            "gyr_T_inv": gyr_res[0],
            "gyr_b": gyr_res[1],
            "gyr_k1": gyr_res[2],
            "gyr_k2": gyr_res[3],
            "gyr_h": gyr_res[4],
            "gyr_params": gyr_res[5],
        }
