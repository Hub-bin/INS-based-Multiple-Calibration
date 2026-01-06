import gtsam
import numpy as np


class OnlineCalibrator:
    """
    Online Calibration 모듈 (using iSAM2)
    - 데이터가 들어올 때마다 Factor Graph를 점진적으로 업데이트합니다.
    - 실시간으로 Bias가 수렴해가는 과정을 볼 수 있습니다.
    """

    def __init__(self, init_bias=None):
        # 1. 초기 파라미터 설정
        if init_bias is None:
            self.init_bias = gtsam.imuBias.ConstantBias()
        else:
            self.init_bias = init_bias

        # IMU 파라미터 (Offline과 동일)
        self.params = gtsam.PreintegrationParams.MakeSharedU(9.81)
        self.params.setAccelerometerCovariance(1e-3 * np.eye(3))
        self.params.setGyroscopeCovariance(1e-4 * np.eye(3))
        self.params.setIntegrationCovariance(1e-4 * np.eye(3))

        # 2. iSAM2 솔버 초기화
        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(0.1)  # 재선형화 임계값
        self.isam = gtsam.ISAM2(parameters)

        # 3. 상태 관리 변수
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimates = gtsam.Values()
        self.step = 0
        self.prev_state = None  # 이전 단계의 (Pose, Velocity) 상태

        # Preintegration 누적 객체
        self.pim = gtsam.PreintegratedImuMeasurements(self.params, self.init_bias)

        # Bias Key는 B(0) 하나만 씁니다. (상수 바이어스 가정)
        self.bias_key = gtsam.symbol("b", 0)

    def initialize(self, start_pose, start_vel):
        """
        첫 프레임에서 Prior Factor를 추가하고 초기화합니다.
        """
        X = lambda i: gtsam.symbol("x", i)
        V = lambda i: gtsam.symbol("v", i)

        # Prior Factors
        self.graph.add(
            gtsam.PriorFactorPose3(X(0), start_pose, gtsam.noiseModel.Isotropic.Sigma(6, 0.001))
        )
        self.graph.add(
            gtsam.PriorFactorVector(V(0), start_vel, gtsam.noiseModel.Isotropic.Sigma(3, 0.01))
        )
        self.graph.add(
            gtsam.PriorFactorConstantBias(
                self.bias_key, self.init_bias, gtsam.noiseModel.Isotropic.Sigma(6, 1.0)
            )
        )

        # Initial Estimates
        self.initial_estimates.insert(X(0), start_pose)
        self.initial_estimates.insert(V(0), start_vel)
        self.initial_estimates.insert(self.bias_key, self.init_bias)

        # iSAM2 Update (초기화)
        self.isam.update(self.graph, self.initial_estimates)

        # 그래프/Estimate 비우기 (iSAM2가 내부적으로 정보를 가짐)
        self.graph.resize(0)
        self.initial_estimates.clear()

        self.prev_state = (start_pose, start_vel)
        self.step = 0

    def update(self, measurement_pose, imu_accel, imu_gyro, dt):
        """
        매 스텝마다 호출되어 그래프를 확장하고 Bias를 추정합니다.

        :param measurement_pose: 현재 시점의 GPS(Pose) 측정값
        :param imu_accel: 현재 시점의 가속도 측정값
        :param imu_gyro: 현재 시점의 자이로 측정값
        :param dt: 시간 간격
        :return: 현재 추정된 Bias
        """
        # 1. IMU Preintegration (적분)
        self.pim.integrateMeasurement(imu_accel, imu_gyro, dt)

        # 다음 스텝 인덱스
        curr_idx = self.step
        next_idx = self.step + 1

        X = lambda i: gtsam.symbol("x", i)
        V = lambda i: gtsam.symbol("v", i)

        # 2. Factor 추가
        # 2-1. IMU Factor (이전 상태 -> 다음 상태 연결)
        factor = gtsam.ImuFactor(
            X(curr_idx), V(curr_idx), X(next_idx), V(next_idx), self.bias_key, self.pim
        )
        self.graph.add(factor)

        # 2-2. GPS Factor (다음 상태에 대한 절대 위치 관측)
        gps_factor = gtsam.PriorFactorPose3(
            X(next_idx), measurement_pose, gtsam.noiseModel.Isotropic.Sigma(6, 0.01)
        )
        self.graph.add(gps_factor)

        # 3. Initial Estimate 추가 (예측값이나 측정값을 초기값으로 사용)
        # 단순화를 위해 측정된 Pose와 이전 속도를 그대로 초기값으로 넣음
        prev_pose, prev_vel = self.prev_state
        self.initial_estimates.insert(X(next_idx), measurement_pose)
        self.initial_estimates.insert(V(next_idx), prev_vel)

        # 4. iSAM2 Update (핵심)
        # 새로운 Factor와 변수만 넣어서 업데이트 수행
        self.isam.update(self.graph, self.initial_estimates)

        # 5. 결과 조회 (현재 추정치 계산)
        result = self.isam.calculateEstimate()
        current_bias = result.atConstantBias(self.bias_key)
        current_vel = result.atVector(V(next_idx))

        # 6. 다음 스텝 준비
        # 그래프와 Estimate 컨테이너를 비워야 함 (iSAM2에 이미 반영됨)
        self.graph.resize(0)
        self.initial_estimates.clear()

        # Preintegration 리셋 (Bias가 바뀌었을 수도 있으므로 현재 Bias로 리셋하는 것이 좋음)
        # 여기서는 연산 속도를 위해 단순 리셋
        self.pim.resetIntegration()

        self.prev_state = (measurement_pose, current_vel)
        self.step += 1

        return current_bias
