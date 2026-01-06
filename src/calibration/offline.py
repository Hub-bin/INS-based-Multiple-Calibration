import gtsam
import numpy as np


class OfflineCalibrator:
    """
    Offline Calibration 모듈
    - 수집된 전체 IMU 데이터와 Ground Truth Pose를 사용하여
    - IMU의 Bias(Accel, Gyro)를 최적화(Estimation)합니다.
    """

    def __init__(self, init_bias=None):
        # 최적화에 사용할 기본 바이어스 (초기 추정값은 보통 0으로 시작)
        if init_bias is None:
            self.init_bias = gtsam.imuBias.ConstantBias()
        else:
            self.init_bias = init_bias

        # IMU Preintegration 파라미터 설정 (노이즈 밀도 등)
        # 실제로는 센서 스펙시트에 있는 값을 넣어야 합니다.
        self.params = gtsam.PreintegrationParams.MakeSharedU(9.81)
        self.params.setAccelerometerCovariance(1e-3 * np.eye(3))  # Accel Noise
        self.params.setGyroscopeCovariance(1e-4 * np.eye(3))  # Gyro Noise
        self.params.setIntegrationCovariance(1e-4 * np.eye(3))  # Integration Error

    def run(self, poses, imu_data, dt):
        """
        Factor Graph를 구축하고 최적화를 수행합니다.

        :param poses: Ground Truth Poses (List of gtsam.Pose3) - GPS 역할
        :param imu_data: IMU Measurements (List of [accel, gyro])
        :param dt: IMU sampling time
        :return: 최적화된 gtsam.imuBias.ConstantBias 객체
        """
        graph = gtsam.NonlinearFactorGraph()
        initial_estimates = gtsam.Values()

        # Key 생성 헬퍼
        X = lambda i: gtsam.symbol("x", i)  # Pose
        V = lambda i: gtsam.symbol("v", i)  # Velocity
        B = lambda i: gtsam.symbol("b", i)  # Bias (우리는 Constant Bias 가정하므로 B(0)만 사용)

        # 1. Prior Factor 추가 (초기 상태 고정)
        # Pose, Velocity, Bias에 대한 초기 불확실성 설정
        start_pose = poses[0]
        start_vel = gtsam.Point3(10.0, 0.0, 0.0)  # 초기 속도 (x=10m/s 가정)

        # Bias는 하나만 추정할 것이므로 B(0) 하나만 그래프에 추가 (Constant Bias Assumption)
        bias_key = B(0)

        graph.add(
            gtsam.PriorFactorPose3(X(0), start_pose, gtsam.noiseModel.Isotropic.Sigma(6, 0.01))
        )
        graph.add(
            gtsam.PriorFactorVector(V(0), start_vel, gtsam.noiseModel.Isotropic.Sigma(3, 0.1))
        )
        graph.add(
            gtsam.PriorFactorConstantBias(
                bias_key, self.init_bias, gtsam.noiseModel.Isotropic.Sigma(6, 1.0)
            )
        )

        initial_estimates.insert(X(0), start_pose)
        initial_estimates.insert(V(0), start_vel)
        initial_estimates.insert(bias_key, self.init_bias)

        # 2. IMU Preintegration 객체 생성
        pim = gtsam.PreintegratedImuMeasurements(self.params, self.init_bias)

        # 3. 데이터 루프 (Graph Building)
        # 간단한 테스트를 위해 매 스텝마다 GPS(Pose)가 들어온다고 가정합니다.
        # 실제로는 GPS가 저주파(1Hz)지만, 여기서는 Bias 관측성을 극대화하기 위해 매 스텝 넣습니다.
        for i in range(len(imu_data) - 1):
            # 3-1. IMU 적분
            accel, gyro = imu_data[i]
            pim.integrateMeasurement(accel, gyro, dt)

            # 3-2. Factor 추가 (ImuFactor)
            # X(i), V(i) -> X(i+1), V(i+1) 사이를 IMU로 연결, Bias는 B(0) 공유
            factor = gtsam.ImuFactor(X(i), V(i), X(i + 1), V(i + 1), bias_key, pim)
            graph.add(factor)

            # 3-3. GPS Factor (Absolute Pose) 추가
            # Bias를 정확히 추정하려면 위치/자세 정보가 강하게 잡혀있어야 함
            current_pose_meas = poses[i + 1]
            gps_factor = gtsam.PriorFactorPose3(
                X(i + 1), current_pose_meas, gtsam.noiseModel.Isotropic.Sigma(6, 0.01)
            )
            graph.add(gps_factor)

            # 3-4. 초기값(Estimate) 추가
            initial_estimates.insert(X(i + 1), poses[i + 1])
            initial_estimates.insert(
                V(i + 1), start_vel
            )  # 속도는 대략 일정하다고 가정 (초기값 용도)

            # Preintegration 리셋
            pim.resetIntegration()

        # 4. 최적화 실행 (Levenberg-Marquardt)
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates)
        result = optimizer.optimize()

        # 5. 추정된 Bias 반환
        estimated_bias = result.atConstantBias(bias_key)
        return estimated_bias
