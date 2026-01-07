import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_comprehensive_dashboard(traj_dict, error_dict, bias_dict, matrix_dict):
    """
    종합 성능 평가 대시보드 시각화
    :param traj_dict: {'GT': poses, 'Conv': poses, 'RL': poses}
    :param error_dict: {'Conv': pos_errors, 'RL': pos_errors} (List of floats)
    :param bias_dict: {'True': (acc, gyr), 'Conv': (acc, gyr), 'RL': (acc, gyr)}
    :param matrix_dict: {'True_T_acc': np.array, 'RL_T_acc_inv': np.array}
    """

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 4, figure=fig)

    # ----------------------------------------------------
    # 1. 궤적 비교 (Top-down View) - 큰 화면 (좌측 상단)
    # ----------------------------------------------------
    ax_traj = fig.add_subplot(gs[0:2, 0:2])

    gt_pos = np.array([[p.x(), p.y()] for p in traj_dict["GT"]])
    conv_pos = np.array([[p.x(), p.y()] for p in traj_dict["Conv"]])
    rl_pos = np.array([[p.x(), p.y()] for p in traj_dict["RL"]])

    ax_traj.plot(gt_pos[:, 0], gt_pos[:, 1], "k--", linewidth=2, label="Ground Truth")
    ax_traj.plot(conv_pos[:, 0], conv_pos[:, 1], "r-", alpha=0.7, label="Conventional (GTSAM only)")
    ax_traj.plot(
        rl_pos[:, 0], rl_pos[:, 1], "b-", linewidth=2, alpha=0.8, label="Proposed (RL + GTSAM)"
    )

    ax_traj.set_title("Trajectory Comparison (2D)", fontsize=14, fontweight="bold")
    ax_traj.set_xlabel("X (m)")
    ax_traj.set_ylabel("Y (m)")
    ax_traj.legend()
    ax_traj.grid(True)
    ax_traj.axis("equal")

    # ----------------------------------------------------
    # 2. 위치 오차 (Position Error over Time) - (우측 상단)
    # ----------------------------------------------------
    ax_err = fig.add_subplot(gs[0, 2:])

    ax_err.plot(error_dict["Conv"], "r-", label=f"Conv (Mean: {np.mean(error_dict['Conv']):.2f}m)")
    ax_err.plot(error_dict["RL"], "b-", label=f"RL (Mean: {np.mean(error_dict['RL']):.2f}m)")

    ax_err.set_title("Position Error Evolution", fontsize=12)
    ax_err.set_xlabel("Time Step")
    ax_err.set_ylabel("Error (m)")
    ax_err.legend()
    ax_err.grid(True)

    # ----------------------------------------------------
    # 3. Bias 추정 성능 비교 (Bar Chart) - (중간 우측)
    # ----------------------------------------------------
    ax_bias_acc = fig.add_subplot(gs[1, 2])
    ax_bias_gyr = fig.add_subplot(gs[1, 3])

    # 데이터 준비
    labels = ["X", "Y", "Z"]
    x = np.arange(len(labels))
    width = 0.25

    # Accel Bias
    ax_bias_acc.bar(x - width, bias_dict["True"][0], width, label="True", color="gray")
    ax_bias_acc.bar(x, bias_dict["Conv"][0], width, label="Conv Est", color="salmon")
    ax_bias_acc.bar(x + width, bias_dict["RL"][0], width, label="RL Est", color="royalblue")
    ax_bias_acc.set_title("Accel Bias Estimation")
    ax_bias_acc.set_xticks(x)
    ax_bias_acc.set_xticklabels(labels)
    ax_bias_acc.grid(axis="y", linestyle="--", alpha=0.5)

    # Gyro Bias
    ax_bias_gyr.bar(x - width, bias_dict["True"][1], width, label="True", color="gray")
    ax_bias_gyr.bar(x, bias_dict["Conv"][1], width, label="Conv Est", color="salmon")
    ax_bias_gyr.bar(x + width, bias_dict["RL"][1], width, label="RL Est", color="royalblue")
    ax_bias_gyr.set_title("Gyro Bias Estimation")
    ax_bias_gyr.set_xticks(x)
    ax_bias_gyr.set_xticklabels(labels)
    ax_bias_gyr.legend()
    ax_bias_gyr.grid(axis="y", linestyle="--", alpha=0.5)

    # ----------------------------------------------------
    # 4. Scale & Misalignment Matrix 분석 - (하단)
    # ----------------------------------------------------
    # RL이 추정한 역행렬(Inv)과 실제 행렬(T)을 곱했을 때 Identity(단위행렬)에 얼마나 가까운지 확인

    true_T = matrix_dict["True_T_acc"]
    rl_inv_T = matrix_dict["RL_T_acc_inv"]

    # 복원된 행렬 (Reconstructed T) = (rl_inv_T)^-1  <- RL이 추정한 T의 역행렬의 역행렬
    # 혹은 보정 효과 확인: T_residual = rl_inv_T @ true_T (이게 Identity여야 함)
    residual_matrix = rl_inv_T @ true_T
    diff_from_identity = residual_matrix - np.eye(3)

    # Heatmap 1: True Error Matrix (정답)
    ax_mat1 = fig.add_subplot(gs[2, 0])
    im1 = ax_mat1.imshow(true_T, cmap="Oranges", vmin=0.9, vmax=1.1)
    ax_mat1.set_title("True Accel Matrix (Scale+Misalign)")
    fig.colorbar(im1, ax=ax_mat1)
    for (j, i), label in np.ndenumerate(true_T):
        ax_mat1.text(i, j, f"{label:.3f}", ha="center", va="center")

    # Heatmap 2: RL Estimated Inverse Matrix
    ax_mat2 = fig.add_subplot(gs[2, 1])
    im2 = ax_mat2.imshow(rl_inv_T, cmap="Blues", vmin=0.9, vmax=1.1)
    ax_mat2.set_title("RL Predicted Correction Matrix (Inv)")
    fig.colorbar(im2, ax=ax_mat2)
    for (j, i), label in np.ndenumerate(rl_inv_T):
        ax_mat2.text(i, j, f"{label:.3f}", ha="center", va="center")

    # Heatmap 3: Correction Result (Should be Identity)
    ax_mat3 = fig.add_subplot(gs[2, 2:])
    im3 = ax_mat3.imshow(residual_matrix, cmap="Greens", vmin=0.95, vmax=1.05)
    ax_mat3.set_title("Resultant Matrix (Pred_Inv * True) -> Should be Identity")
    fig.colorbar(im3, ax=ax_mat3)
    for (j, i), label in np.ndenumerate(residual_matrix):
        color = "white" if abs(label - (1 if i == j else 0)) > 0.01 else "black"
        ax_mat3.text(i, j, f"{label:.4f}", ha="center", va="center", color=color)

    plt.tight_layout()
    plt.show()
