import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.utils.road_generator import RoadTrajectoryGenerator


def verify_physics():
    print("=== Physics Sanity Check ===")
    road_gen = RoadTrajectoryGenerator(location_point=(35.1796, 129.0756), dist=1000)

    # 경로 생성 및 보간
    x_pts, y_pts, _ = road_gen.generate_path()
    dt = 0.1
    traj_data = road_gen.interpolate_trajectory(x_pts, y_pts, target_speed=15.0, dt=dt)

    # 데이터 추출
    accels = np.array([d["accel_body"] for d in traj_data])
    omegas = np.array([d["omega_body"] for d in traj_data])
    vels = np.array([d["vel_body"] for d in traj_data])

    # 1. 속도 검증 (Target Speed 15m/s 근처여야 함)
    # Body Frame에서 x축 속도가 주행 속도여야 함
    mean_vx = np.mean(vels[:, 0])
    print(f"[Check 1] Mean Speed (Body X): {mean_vx:.2f} m/s (Target: 15.0)")

    # 2. 가속도 검증 (일반 승용차는 급가속해도 0.5g ~ 5m/s^2를 넘기 힘듦)
    max_acc = np.max(np.abs(accels))
    print(f"[Check 2] Max Acceleration: {max_acc:.2f} m/s^2")

    if max_acc > 10.0:
        print("  -> WARNING: Acceleration is too high! (Unrealistic physics)")
    else:
        print("  -> Acceleration is within realistic range.")

    # 3. 각속도 검증 (일반 도로 커브에서 1 rad/s = 57도/s 이상 돌기 힘듦)
    max_omega = np.max(np.abs(omegas))
    print(f"[Check 3] Max Turn Rate: {max_omega:.2f} rad/s")

    if max_omega > 1.5:
        print("  -> WARNING: Turning too fast!")
    else:
        print("  -> Turn rate is realistic.")


if __name__ == "__main__":
    verify_physics()
