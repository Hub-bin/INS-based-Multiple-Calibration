import numpy as np
from scipy.optimize import minimize


class SysIdCalibrator:
    """
    수치 최적화(Optimization) 기반의 시스템 식별 교정기
    - 학습(Training)이 아닌 최적화(Solving)를 통해 파라미터를 찾습니다.
    - 모델: Measured = T * True + Bias
    - 목표: Minimize || Measured - (T * True + Bias) ||^2
    """

    def __init__(self):
        pass

    def calibrate_sensor(self, true_data, raw_data, sensor_type="accel"):
        """
        :param true_data: (N, 3) Ground Truth (Ideal)
        :param raw_data: (N, 3) Measured (Error included)
        """
        # 최적화할 파라미터: 12개 (9 Matrix elements + 3 Bias elements)
        # 초기값: T=Identity, Bias=0
        initial_guess = np.concatenate([np.eye(3).flatten(), np.zeros(3)])

        # Loss Function
        def loss_func(params):
            # 파라미터 복원
            T = params[:9].reshape(3, 3)
            b = params[9:]

            # 예측값 계산: Pred = T * True + b
            # (N,3) = (N,3) @ T.T + b
            pred = true_data @ T.T + b

            # MSE 계산
            diff = raw_data - pred
            return np.sum(diff**2)

        # 최적화 실행 (L-BFGS-B 알고리즘 등 사용)
        print(f"Running Optimization for {sensor_type}...")
        result = minimize(loss_func, initial_guess, method="L-BFGS-B")

        if result.success:
            print(f"  -> Converged! Loss: {result.fun:.4f}")
            opt_T = result.x[:9].reshape(3, 3)
            opt_b = result.x[9:]

            # 우리가 원하는 건 Correction 모델임
            # Raw = T * True + b
            # True = T_inv * (Raw - b)
            # Correction Matrix = T_inv, Correction Bias = b

            try:
                T_inv = np.linalg.inv(opt_T)
            except np.linalg.LinAlgError:
                print("  -> Matrix inversion failed. Using Identity.")
                T_inv = np.eye(3)

            return T_inv, opt_b
        else:
            print("  -> Optimization Failed.")
            return np.eye(3), np.zeros(3)

    def run(self, true_measurements, raw_measurements):
        """
        :param true_measurements: List of (acc, gyr)
        :param raw_measurements: List of (acc, gyr)
        """
        # 데이터 포맷 변환 (List -> Numpy Array)
        true_acc = np.array([m[0] for m in true_measurements])
        true_gyr = np.array([m[1] for m in true_measurements])

        raw_acc = np.array([m[0] for m in raw_measurements])
        raw_gyr = np.array([m[1] for m in raw_measurements])

        # 가속도계 교정
        acc_T_inv, acc_b = self.calibrate_sensor(true_acc, raw_acc, "Accel")

        # 자이로스코프 교정
        gyr_T_inv, gyr_b = self.calibrate_sensor(true_gyr, raw_gyr, "Gyro")

        return {"acc_T_inv": acc_T_inv, "acc_b": acc_b, "gyr_T_inv": gyr_T_inv, "gyr_b": gyr_b}

    def correct(self, raw_measurements, params):
        corrected = []

        T_acc = params["acc_T_inv"]
        b_acc = params["acc_b"]
        T_gyr = params["gyr_T_inv"]
        b_gyr = params["gyr_b"]

        for ra, rg in raw_measurements:
            # Corrected = T_inv * (Raw - Bias)
            ca = T_acc @ (ra - b_acc)
            cg = T_gyr @ (rg - b_gyr)
            corrected.append((ca, cg))

        return corrected
