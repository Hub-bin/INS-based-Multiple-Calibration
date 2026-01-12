import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gtsam
from src.utils.road_generator import RoadTrajectoryGenerator
from src.simulation.profile import TrajectorySimulator
from src.navigation.strapdown import StrapdownNavigator
from src.sensors.imu import ImuSensor


def verify_navigation():
    print("\n>>> [Step 2] 항법 엔진 무결성 검증 (Navigation Engine Integrity)")

    dt = 0.1
    road_gen = RoadTrajectoryGenerator(location_point=(35.1796, 129.0756), dist=2000)
    sim = TrajectorySimulator(road_gen, dt)
    traj_data = sim.generate_3d_profile(total_duration_min=3)

    imu = ImuSensor(
        accel_bias=[0, 0, 0],
        accel_hysteresis=[0, 0, 0],
        accel_noise=0.0,
        gyro_bias=[0, 0, 0],
        gyro_noise=0.0,
    )

    start_pose = traj_data[0]["pose"]
    start_vel = traj_data[0]["vel_world"]

    nav = StrapdownNavigator(start_pose, gravity=9.81)
    nav.curr_vel = start_vel

    final_err = 0.0

    for i, data in enumerate(traj_data):
        meas_acc, meas_gyr, _ = imu.measure(
            data["pose"], data["sf_true"], data["omega_body"], data["temp"]
        )

        nav.integrate(meas_acc, meas_gyr, dt)
        pred_pose = nav.predict()

        # [수정] 안전한 좌표 추출
        gt_p_obj = data["pose"].translation()
        est_p_obj = pred_pose.translation()

        if hasattr(gt_p_obj, "x"):
            p_gt = np.array([gt_p_obj.x(), gt_p_obj.y(), gt_p_obj.z()])
        else:
            p_gt = gt_p_obj

        if hasattr(est_p_obj, "x"):
            p_est = np.array([est_p_obj.x(), est_p_obj.y(), est_p_obj.z()])
        else:
            p_est = est_p_obj

        dist = np.linalg.norm(p_gt - p_est)
        final_err = dist

    print(f"  > Final Navigation Error (Zero Noise): {final_err:.4f} m")

    if final_err < 5.0:
        print("  ✅ [PASS] 항법 엔진이 정상 작동합니다.")
    else:
        print("  ❌ [FAIL] 항법 엔진 설정에 문제가 있습니다.")


if __name__ == "__main__":
    verify_navigation()
