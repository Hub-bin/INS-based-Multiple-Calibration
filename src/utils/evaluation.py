import numpy as np
import matplotlib.pyplot as plt
import gtsam


def calculate_rmse(gt_poses, est_poses):
    """Ground Truth와 추정 궤적 간의 RMSE 계산 (Position & Rotation)"""
    n = min(len(gt_poses), len(est_poses))
    sq_err_pos = 0.0
    sq_err_rot = 0.0

    for i in range(n):
        error_pose = gt_poses[i].between(est_poses[i])
        sq_err_pos += np.sum(error_pose.translation() ** 2)
        sq_err_rot += np.sum(error_pose.rotation().xyz() ** 2)

    rmse_pos = np.sqrt(sq_err_pos / n)
    rmse_rot = np.sqrt(sq_err_rot / n)
    return rmse_pos, rmse_rot


def plot_trajectory_comparison(gt_poses, est_poses, title="Trajectory Comparison"):
    """GT와 Estimated 궤적 비교 Plot (2D Top-down View)"""
    gt_x = [p.x() for p in gt_poses]
    gt_y = [p.y() for p in gt_poses]

    est_x = [p.x() for p in est_poses]
    est_y = [p.y() for p in est_poses]

    plt.figure(figsize=(10, 6))
    plt.plot(gt_x, gt_y, "k--", label="Ground Truth", linewidth=2)
    plt.plot(est_x, est_y, "b-", label="Estimated", alpha=0.7)
    plt.title(title)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def plot_error_analysis(gt_poses, est_poses):
    """시간에 따른 위치 오차 그래프"""
    n = min(len(gt_poses), len(est_poses))
    errors = []
    for i in range(n):
        pos_diff = gt_poses[i].translation() - est_poses[i].translation()
        errors.append(np.linalg.norm(pos_diff))

    plt.figure(figsize=(10, 4))
    plt.plot(errors, "r-", label="Position Error")
    plt.title("Navigation Position Error over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Error (m)")
    plt.legend()
    plt.grid(True)
    plt.show()
