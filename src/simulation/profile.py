import numpy as np
import gtsam


class TrajectorySimulator:
    def __init__(self, road_generator, dt=0.1):
        self.road_gen = road_generator
        self.dt = dt

    def generate_3d_profile(self, total_duration_min=10):
        """
        일반 주행 시나리오 (항법 성능 테스트용)
        - 운동학적 일관성(Kinematic Consistency) 보장: 위치 미분 -> 속도 -> 가속도
        """
        total_steps = int(total_duration_min * 60 / self.dt)

        # 1. Base Trajectory Generation (Position & Heading)
        path_x, path_y, _ = self.road_gen.generate_path()
        trajectory = self.road_gen.interpolate_trajectory(
            path_x, path_y, target_speed=20.0, dt=self.dt
        )
        n_points = len(trajectory)

        sim_data = []
        poses = []
        vels_world = []  # v = dp/dt
        accs_world = []  # a = dv/dt

        curr_idx = 0.0
        curr_speed = 0.0

        # print(f"[Sim] Generating kinematic consistent profile ({total_duration_min} mins)...")

        # --- Phase 1: Generate Pose Trajectory first ---
        for i in range(total_steps + 2):  # Extra points for differentiation
            t = i * self.dt

            # Speed Profile
            traffic_phase = int(t / 60.0) % 5
            if traffic_phase == 0 or traffic_phase == 2:
                target_speed = 15.0 + 10.0 * np.sin(2.0 * np.pi * t / 15.0)
                if target_speed < 0:
                    target_speed = 0
            elif traffic_phase == 4:
                target_speed = 0.0
            else:
                target_speed = 25.0 + np.random.normal(0, 0.5)

            acc_lin = (target_speed - curr_speed) / self.dt
            acc_lin = np.clip(acc_lin, -4.0, 3.5)
            curr_speed += acc_lin * self.dt
            if curr_speed < 0:
                curr_speed = 0

            # Update Position Index
            step_dist = curr_speed * self.dt
            curr_idx += step_dist / 2.0
            idx_int = int(min(curr_idx, n_points - 1))

            # Base Pose
            base_pose = trajectory[idx_int]["pose"]
            base_rot = base_pose.rotation()

            # Add Road Dynamics (Roll/Pitch)
            sim_roll = 0.1 * np.sin(2.0 * np.pi * t / 20.0)
            sim_pitch = 0.05 * np.sin(2.0 * np.pi * t / 40.0)

            delta_rot = gtsam.Rot3.Ypr(0, sim_pitch, sim_roll)
            new_rot = base_rot.compose(delta_rot)
            new_pose = gtsam.Pose3(new_rot, base_pose.translation())

            poses.append(new_pose)

        # --- Phase 2: Derive Velocity & Acceleration (Numerical Differentiation) ---
        for i in range(len(poses) - 1):
            p1 = poses[i].translation()
            p2 = poses[i + 1].translation()

            v1 = np.array([p1.x(), p1.y(), p1.z()]) if hasattr(p1, "x") else p1
            v2 = np.array([p2.x(), p2.y(), p2.z()]) if hasattr(p2, "x") else p2

            vel = (v2 - v1) / self.dt
            vels_world.append(vel)

        for i in range(len(vels_world) - 1):
            a_world = (vels_world[i + 1] - vels_world[i]) / self.dt
            accs_world.append(a_world)

        # --- Phase 3: Compute Body Frame Values ---
        for i in range(min(total_steps, len(accs_world))):
            t = i * self.dt
            pose = poses[i]
            rot = pose.rotation()
            vel_w = vels_world[i]
            acc_w = accs_world[i]

            # 1. Body Acceleration (Kinematic)
            acc_w_pt = gtsam.Point3(acc_w[0], acc_w[1], acc_w[2])
            acc_body_pt = rot.unrotate(acc_w_pt)

            if hasattr(acc_body_pt, "x"):
                acc_body = np.array([acc_body_pt.x(), acc_body_pt.y(), acc_body_pt.z()])
            else:
                acc_body = acc_body_pt

            # 2. Body Angular Velocity (Omega)
            if i < len(poses) - 1:
                R_curr = poses[i].rotation()
                R_next = poses[i + 1].rotation()
                dRot = R_curr.between(R_next)
                omega_body = gtsam.Rot3.Logmap(dRot) / self.dt
            else:
                omega_body = np.zeros(3)

            # 3. True Specific Force (Gravity Removal)
            g_world_pt = gtsam.Point3(0, 0, -9.81)
            g_body_pt = rot.unrotate(g_world_pt)

            if hasattr(g_body_pt, "x"):
                g_vec = np.array([g_body_pt.x(), g_body_pt.y(), g_body_pt.z()])
            else:
                g_vec = g_body_pt

            sf_true = acc_body - g_vec
            speed = np.linalg.norm(vel_w)

            sim_data.append(
                {
                    "time": t,
                    "pose": pose,
                    "vel_world": vel_w,
                    "speed": speed,
                    "acc_body": acc_body,
                    "omega_body": omega_body,
                    "sf_true": sf_true,
                    "g_vec": g_vec,
                    "temp": 20.0 + (30.0 * (i / total_steps)),
                }
            )

        return sim_data

    def generate_excitation_profile(self, total_duration_min=10):
        """
        캘리브레이션 정밀도 검증을 위한 '가혹 기동(Excitation)' 프로파일
        - 지속적인 가감속 (X축 Scale/Bias 분리)
        - 지속적인 S자 주행 (Y축 Scale/Bias 분리)
        - 지속적인 펌프/경사 (Z축 Scale/Bias 분리)
        """
        total_steps = int(total_duration_min * 60 / self.dt)
        path_x, path_y, _ = self.road_gen.generate_path()
        trajectory = self.road_gen.interpolate_trajectory(
            path_x, path_y, target_speed=20.0, dt=self.dt
        )
        n_points = len(trajectory)

        sim_data = []
        curr_idx = 0.0
        curr_speed = 0.0
        prev_pos = None

        print(f"[Sim] Generating EXCITATION profile ({total_duration_min} mins)...")

        for i in range(total_steps):
            t = i * self.dt

            # 1. Dynamic Speed Profile (급가감속 반복)
            target_speed = 20.0 + 10.0 * np.sin(2.0 * np.pi * t / 20.0)
            acc_lin = (target_speed - curr_speed) / self.dt
            acc_lin = np.clip(acc_lin, -8.0, 6.0)

            curr_speed += acc_lin * self.dt
            if curr_speed < 0:
                curr_speed = 0

            # Position Update
            step_dist = curr_speed * self.dt
            curr_idx += step_dist / 2.0
            idx_int = int(min(curr_idx, n_points - 1))

            # 2. Dynamic Attitude (롤링/피칭 극대화)
            base_pose = trajectory[idx_int]["pose"]
            base_rot = base_pose.rotation()

            sim_roll = 0.17 * np.sin(2.0 * np.pi * t / 15.0)
            sim_pitch = 0.08 * np.sin(2.0 * np.pi * t / 25.0)

            delta_rot = gtsam.Rot3.Ypr(0, sim_pitch, sim_roll)
            new_rot = base_rot.compose(delta_rot)
            new_pose = gtsam.Pose3(new_rot, base_pose.translation())

            # 3. Kinematic Consistent Velocity (Approximate for Excitation)
            # 여기서는 정밀 항법보다는 센서 자극이 목적이므로 약식 계산 허용하되 벡터 유지
            curr_trans = new_pose.translation()
            curr_pos_vec = (
                np.array([curr_trans.x(), curr_trans.y(), curr_trans.z()])
                if hasattr(curr_trans, "x")
                else curr_trans
            )

            if prev_pos is None:
                R0 = new_rot.matrix()
                vel_world = R0 @ np.array([curr_speed, 0, 0])
            else:
                vel_world = (curr_pos_vec - prev_pos) / self.dt
            prev_pos = curr_pos_vec

            # 4. Body Acceleration (High Dynamics)
            # 인위적인 진동 추가 (엔진 떨림 등 모사)
            vib_x = 0.2 * np.random.randn()
            vib_y = 1.5 * np.sin(2.0 * np.pi * t / 5.0)
            vib_z = 1.0 * np.sin(2.0 * np.pi * t / 3.0)

            acc_body = np.array([acc_lin + vib_x, vib_y, vib_z])

            # Omega (High Rotational Dynamics)
            omega = np.array(
                [
                    0.2 * np.cos(2 * np.pi * t / 15.0),
                    0.1 * np.cos(2 * np.pi * t / 25.0),
                    0.3 * np.sin(2 * np.pi * t / 10.0),
                ]
            )

            # 5. True Specific Force
            g_body_obj = new_rot.unrotate(gtsam.Point3(0, 0, -9.81))
            g_vec = (
                np.array([g_body_obj.x(), g_body_obj.y(), g_body_obj.z()])
                if hasattr(g_body_obj, "x")
                else g_body_obj
            )

            sf_true = acc_body - g_vec

            sim_data.append(
                {
                    "time": t,
                    "pose": new_pose,
                    "vel_world": vel_world,
                    "speed": curr_speed,
                    "acc_body": acc_body,
                    "omega_body": omega,
                    "sf_true": sf_true,
                    "g_vec": g_vec,
                    "temp": 20.0 + (30.0 * (i / total_steps)),
                }
            )

        return sim_data
