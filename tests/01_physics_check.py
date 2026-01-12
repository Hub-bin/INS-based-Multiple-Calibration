import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import gtsam

from src.utils.road_generator import RoadTrajectoryGenerator
from src.simulation.profile import TrajectorySimulator


def verify_physics():
    print(">>> [Step 1] 물리 정합성 검증 (Physics Consistency Check)")

    dt = 0.1
    road_gen = RoadTrajectoryGenerator(location_point=(35.1796, 129.0756), dist=2000)
    sim = TrajectorySimulator(road_gen, dt)

    # 3분간 동적 주행 데이터 생성
    traj_data = sim.generate_3d_profile(total_duration_min=3)

    # 초기 상태
    init_p = traj_data[0]["pose"].translation()
    if hasattr(init_p, "x"):
        calc_pos = np.array([init_p.x(), init_p.y(), init_p.z()])
    else:
        calc_pos = init_p

    calc_vel = traj_data[0]["vel_world"]

    pos_errors = []

    for i in range(len(traj_data) - 1):
        curr_data = traj_data[i]
        rot = curr_data["pose"].rotation()
        sf_body = curr_data["sf_true"]

        # [수정] 안전한 벡터 추출
        # R * sf (Body Specific Force -> World Frame)
        acc_kine_obj = rot.rotate(gtsam.Point3(*sf_body))
        if hasattr(acc_kine_obj, "vector"):
            acc_kine_world = acc_kine_obj.vector()
        elif hasattr(acc_kine_obj, "x"):  # Point3 but no .vector()
            acc_kine_world = np.array([acc_kine_obj.x(), acc_kine_obj.y(), acc_kine_obj.z()])
        else:  # Already numpy array
            acc_kine_world = acc_kine_obj

        # True Acc = Kinematic Acc + Gravity
        # sf = a - g  =>  a = sf + g
        # World Frame: a_world = (R * sf_body) + g_world
        g_world = np.array([0, 0, -9.81])
        acc_world = acc_kine_world + g_world

        # Euler Integration
        calc_pos += calc_vel * dt + 0.5 * acc_world * dt**2
        calc_vel += acc_world * dt

        # Compare with Ground Truth
        gt_next_pose = traj_data[i + 1]["pose"].translation()
        if hasattr(gt_next_pose, "x"):
            gt_pos = np.array([gt_next_pose.x(), gt_next_pose.y(), gt_next_pose.z()])
        else:
            gt_pos = gt_next_pose

        err = np.linalg.norm(gt_pos - calc_pos)
        pos_errors.append(err)

    final_err = pos_errors[-1]
    print(f"  > Final Position Integration Error: {final_err:.4f} m")

    if final_err < 10.0:
        print("  ✅ [PASS] 물리 데이터가 정합성을 가집니다.")
    else:
        print("  ❌ [FAIL] 시뮬레이션 데이터 자체에 모순이 있습니다.")

    plt.figure()
    plt.plot(pos_errors)
    plt.title("Integration Error over Time (Manual Integration)")
    plt.ylabel("Error (m)")
    plt.xlabel("Step")
    plt.grid()
    plt.savefig("tests/01_physics_result.png")
    print("  > Result saved to tests/01_physics_result.png")


if __name__ == "__main__":
    if not os.path.exists("tests"):
        os.makedirs("tests")
    verify_physics()
