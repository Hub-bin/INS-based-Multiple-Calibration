import numpy as np
import gtsam


class TrajectorySimulator:
    def __init__(self, road_generator, dt=0.1):
        self.road_gen = road_generator
        self.dt = dt

    def generate_3d_profile(self, total_duration_min=10):
        total_steps = int(total_duration_min * 60 / self.dt)
        trajectory = self.road_gen.interpolate_trajectory(
            *self.road_gen.generate_path()[:2], target_speed=20.0, dt=self.dt
        )
        n_points = len(trajectory)

        sim_data = []
        curr_idx = 0.0
        curr_vel = 0.0

        print(f"[Sim] Generating {total_duration_min} mins profile with 3D dynamics...")

        for i in range(total_steps):
            t = i * self.dt

            # 1. 속도 프로파일
            traffic_phase = int(t / 60.0) % 5
            if traffic_phase == 0 or traffic_phase == 2:
                target_vel = 15.0 + 10.0 * np.sin(2.0 * np.pi * t / 15.0)
                if target_vel < 0:
                    target_vel = 0
            elif traffic_phase == 4:
                target_vel = 0.0
            else:
                target_vel = 25.0 + np.random.normal(0, 0.5)

            acc_lin = target_vel - curr_vel
            acc_lin = np.clip(acc_lin, -4.0, 3.5)
            curr_vel += acc_lin * self.dt
            if curr_vel < 0:
                curr_vel = 0

            # 위치 업데이트
            step_dist = curr_vel * self.dt
            curr_idx += step_dist / 2.0
            idx_int = int(min(curr_idx, n_points - 1))

            # 2. 자세(Attitude) 생성
            base_pose = trajectory[idx_int]["pose"]
            base_rot = base_pose.rotation()

            sim_roll = 0.1 * np.sin(2.0 * np.pi * t / 20.0)
            sim_pitch = 0.05 * np.sin(2.0 * np.pi * t / 40.0)

            delta_rot = gtsam.Rot3.Ypr(0, sim_pitch, sim_roll)
            new_rot = base_rot.compose(delta_rot)
            new_pose = gtsam.Pose3(new_rot, base_pose.translation())

            # 3. 3축 가속도/각속도 생성
            acc_x = acc_lin
            acc_y = 0.8 * np.sin(2.0 * np.pi * t / 7.0)
            acc_z = 0.5 * np.sin(2.0 * np.pi * t / 3.0)
            acc_body = np.array([acc_x, acc_y, acc_z])

            omega = trajectory[idx_int]["omega_body"] * (curr_vel / 20.0)
            omega += np.array(
                [0.1 * np.cos(2 * np.pi * t / 20.0), 0.05 * np.cos(2 * np.pi * t / 40.0), 0.0]
            )

            # 4. True Specific Force 계산 (중력 제거)
            # 여기가 핵심: 시뮬레이터가 미리 올바른 '비력'을 계산해줌
            g_body_vec = new_rot.unrotate(gtsam.Point3(0, 0, -9.81))
            if hasattr(g_body_vec, "x"):
                g_vec = np.array([g_body_vec.x(), g_body_vec.y(), g_body_vec.z()])
            else:
                g_vec = g_body_vec

            sf_true = acc_body - g_vec

            sim_data.append(
                {
                    "time": t,
                    "pose": new_pose,
                    "vel_world": curr_vel,
                    "acc_body": acc_body,  # 순수 가속도 (참값)
                    "omega_body": omega,  # 순수 각속도 (참값)
                    "sf_true": sf_true,  # 비력 (IMU 입력용 참값)
                    "g_vec": g_vec,  # 중력 벡터 (디버깅용)
                    "temp": 20.0 + (30.0 * (i / total_steps)),
                }
            )

        return sim_data
