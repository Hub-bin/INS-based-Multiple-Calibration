import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src.simulation.profile import TrajectorySimulator
from src.navigation.strapdown import StrapdownNavigator
from src.sensors.imu import ImuSensor


class CalibrationEnv(gym.Env):
    def __init__(self, road_gen, dt=0.1, window_size=600):
        super(CalibrationEnv, self).__init__()
        self.road_gen = road_gen
        self.dt = dt
        self.window_size = window_size

        # [수정] Action: Acc Bias(3) + Gyr Bias(3) = 6개
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # Observation: [Acc, Gyr, Temp]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size, 7), dtype=np.float32
        )

        self.sim = None
        self.traj_data = None
        self.imu = None
        self.temp_coeffs = None
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim = TrajectorySimulator(self.road_gen, self.dt)
        self.traj_data = self.sim.generate_3d_profile(total_duration_min=3)

        # [핵심] 고정된 물리 법칙 (Fixed Physics for Learnability)
        # 온도에 따른 바이어스 변화 규칙을 고정
        self.temp_coeffs = {
            "acc_lin": np.array([0.001, 0.001, 0.001]),  # 1도당 1mg
            "acc_quad": np.array([0.0, 0.0, 0.0]),
            "gyr_lin": np.array([0.00001, 0.00001, 0.00001]),
        }

        # 기본 센서 오차 없음 (온도 효과만 주입)
        self.true_params = {"acc_scale": np.ones(3), "gyr_scale": np.ones(3)}

        self.imu = ImuSensor(
            accel_bias=np.zeros(3),
            accel_hysteresis=np.zeros(3),
            accel_noise=1e-5,
            gyro_bias=np.zeros(3),
            gyro_noise=1e-6,
        )

        self.current_step = self.window_size
        return self._get_observation(self.current_step), {}

    def step(self, action):
        est_bias = self._decode_action(action)

        eval_len = 50
        if self.current_step + eval_len >= len(self.traj_data):
            terminated = True
            eval_len = len(self.traj_data) - self.current_step
        else:
            terminated = False

        if eval_len <= 0:
            return self._get_observation(self.current_step), 0.0, True, False, {}

        start_data = self.traj_data[self.current_step]
        nav = StrapdownNavigator(start_data["pose"], gravity=9.81)
        nav.curr_vel = start_data["vel_world"]

        total_vel_err = 0.0

        for i in range(eval_len):
            idx = self.current_step + i
            data = self.traj_data[idx]

            # 1. 측정
            meas_acc, meas_gyr, _ = self.imu.measure(
                data["pose"], data["sf_true"], data["omega_body"], data["temp"]
            )

            # 2. [물리] 온도 오차 주입 (고정 법칙)
            dt_temp = data["temp"] - 20.0
            meas_acc += self.temp_coeffs["acc_lin"] * dt_temp
            meas_gyr += self.temp_coeffs["gyr_lin"] * dt_temp

            # 3. [보정] Agent Action 적용
            corr_acc = meas_acc - est_bias["acc_bias"]
            corr_gyr = meas_gyr - est_bias["gyr_bias"]

            # 4. 적분
            nav.integrate(corr_acc, corr_gyr, self.dt)
            nav.predict()

            total_vel_err += np.linalg.norm(data["vel_world"] - nav.curr_vel)

        self.current_step += eval_len
        mean_vel_err = total_vel_err / eval_len

        # 보상: 속도 오차 최소화
        reward = -(mean_vel_err * 10.0)

        return self._get_observation(self.current_step), reward, terminated, False, {}

    def _get_observation(self, idx):
        start = idx - self.window_size
        end = idx
        obs_rows = []
        for i in range(start, end):
            if i < 0:
                obs_rows.append(np.zeros(7))
                continue
            data = self.traj_data[i]
            ma, mg, _ = self.imu.measure(
                data["pose"], data["sf_true"], data["omega_body"], data["temp"]
            )

            # 관측값에도 온도 오차 반영
            dt_temp = data["temp"] - 20.0
            ma += self.temp_coeffs["acc_lin"] * dt_temp
            mg += self.temp_coeffs["gyr_lin"] * dt_temp

            # [중요] 정규화 (Normalization)
            row = np.concatenate([ma / 9.81, mg, [(data["temp"] - 20) / 30.0]])
            obs_rows.append(row)
        return np.array(obs_rows, dtype=np.float32)

    def _decode_action(self, action):
        # Action -> Bias Only (Scale/Hyst 제외)
        return {
            "acc_bias": action[0:3] * 0.05,  # +/- 50mg
            "gyr_bias": action[3:6] * 0.005,  # +/- 0.005 rad/s
        }
