import gtsam
import numpy as np


class StrapdownNavigator:
    def __init__(self, start_pose, gravity=9.81):
        # [수정] 표준 INS 방식: PIM이 중력 보정을 수행하도록 설정
        # MakeSharedU(g)는 Z-up 좌표계에서 gravity vector를 (0, 0, -g)로 설정함.
        # 시뮬레이터도 (0, 0, -9.81)을 사용하므로 일치함.
        self.params = gtsam.PreintegrationParams.MakeSharedU(gravity)
        self.bias = gtsam.imuBias.ConstantBias()
        self.pim = gtsam.PreintegratedImuMeasurements(self.params, self.bias)

        self.curr_pose = start_pose
        self.curr_vel = np.zeros(3)
        self.poses = [start_pose]

    def integrate(self, acc, gyr, dt):
        # acc: Specific Force (가속도계 출력)
        # gyr: Angular Velocity (자이로 출력)
        self.pim.integrateMeasurement(acc, gyr, dt)

    def predict(self):
        nav_state = gtsam.NavState(self.curr_pose, self.curr_vel)
        next_state = self.pim.predict(nav_state, gtsam.imuBias.ConstantBias())

        self.curr_pose = next_state.pose()
        self.curr_vel = next_state.velocity()
        self.pim.resetIntegration()

        self.poses.append(self.curr_pose)
        return self.curr_pose

    def zero_velocity_update(self):
        self.curr_vel = np.zeros(3)
