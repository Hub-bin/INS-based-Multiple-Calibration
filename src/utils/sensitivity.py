import numpy as np
import gtsam
import copy


class SensitivityAnalyzer:
    """
    오차 파라미터(Bias, Scale, Misalignment)가 항법 궤적에 미치는 민감도를 분석하는 클래스.
    """

    def __init__(self, vehicle, imu_sensor, dt):
        self.vehicle = vehicle
        self.imu = imu_sensor
        self.dt = dt

    def run_navigation(self, measurements, bias, error_matrix_acc, error_matrix_gyr):
        """주어진 파라미터로 궤적 적분 (Dead Reckoning)"""
        pim_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
        pim = gtsam.PreintegratedImuMeasurements(pim_params, bias)

        poses = [self.vehicle.poses[0]]
        curr_pose = self.vehicle.poses[0]
        curr_vel = gtsam.Point3(10.0, 0, 0)  # 초기 속도

        # 역보정 행렬 준비
        try:
            acc_T_inv = np.linalg.inv(error_matrix_acc)
            gyr_T_inv = np.linalg.inv(error_matrix_gyr)
        except:
            acc_T_inv = np.eye(3)
            gyr_T_inv = np.eye(3)

        for raw_acc, raw_gyr in measurements:
            # 파라미터 섭동의 영향을 보기 위해 보정 후 적분
            acc = acc_T_inv @ (raw_acc - bias.accelerometer())
            gyr = gyr_T_inv @ (raw_gyr - bias.gyroscope())

            pim.integrateMeasurement(acc, gyr, self.dt)

            nav_state = gtsam.NavState(curr_pose, curr_vel)
            next_state = pim.predict(nav_state, bias)
            curr_pose = next_state.pose()
            curr_vel = next_state.velocity()
            poses.append(curr_pose)
            pim.resetIntegration()

        return poses

    def compute_sensitivity(self, raw_measurements, base_params):
        """
        Bias(6) + Scale(6) + Misalignment(12) = 총 24개 파라미터의 민감도 분석
        """
        # Base Trajectory
        base_bias = gtsam.imuBias.ConstantBias(base_params["acc_b"], base_params["gyr_b"])
        base_poses = self.run_navigation(
            raw_measurements, base_bias, base_params["acc_T"], base_params["gyr_T"]
        )
        final_pos_base = base_poses[-1].translation()

        sensitivity_score = {}
        epsilon = 1e-4  # 섭동 크기
        axis_names = ["x", "y", "z"]

        # 1. Bias Sensitivity
        for i, axis in enumerate(axis_names):
            # Accel Bias
            pb = base_params["acc_b"].copy()
            pb[i] += epsilon
            p_bias = gtsam.imuBias.ConstantBias(pb, base_params["gyr_b"])
            p_poses = self.run_navigation(
                raw_measurements, p_bias, base_params["acc_T"], base_params["gyr_T"]
            )
            diff = np.linalg.norm(p_poses[-1].translation() - final_pos_base)
            sensitivity_score[f"Acc_Bias_{axis}"] = diff / epsilon

            # Gyro Bias
            pb = base_params["gyr_b"].copy()
            pb[i] += epsilon
            p_bias = gtsam.imuBias.ConstantBias(base_params["acc_b"], pb)
            p_poses = self.run_navigation(
                raw_measurements, p_bias, base_params["acc_T"], base_params["gyr_T"]
            )
            diff = np.linalg.norm(p_poses[-1].translation() - final_pos_base)
            sensitivity_score[f"Gyr_Bias_{axis}"] = diff / epsilon

        # 2. Matrix (Scale & Misalignment) Sensitivity
        # T[i, j] -> i: sensor axis, j: true axis (Cross-axis effect)
        for i in range(3):
            for j in range(3):
                name_suffix = f"{axis_names[i]}{axis_names[j]}"  # xx, xy, xz ...
                type_name = "Scale" if i == j else "Misalign"

                # Accel Matrix
                pT = base_params["acc_T"].copy()
                pT[i, j] += epsilon
                p_poses = self.run_navigation(raw_measurements, base_bias, pT, base_params["gyr_T"])
                diff = np.linalg.norm(p_poses[-1].translation() - final_pos_base)
                sensitivity_score[f"Acc_{type_name}_{name_suffix}"] = diff / epsilon

                # Gyro Matrix
                pT = base_params["gyr_T"].copy()
                pT[i, j] += epsilon
                p_poses = self.run_navigation(raw_measurements, base_bias, base_params["acc_T"], pT)
                diff = np.linalg.norm(p_poses[-1].translation() - final_pos_base)
                sensitivity_score[f"Gyr_{type_name}_{name_suffix}"] = diff / epsilon

        return sensitivity_score
