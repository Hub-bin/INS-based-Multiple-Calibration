import sys
import os
import numpy as np
import torch
import torch.nn as nn
import gtsam
import matplotlib.pyplot as plt
import folium

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.road_generator import RoadTrajectoryGenerator
from src.sensors.imu import ImuSensor
from src.simulation.profile import TrajectorySimulator


# ==============================================================================
# 1. Network Structure (Must Match Training Code Exactly)
# ==============================================================================
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        nn.init.constant_(m.bias, 0.0)


class FastActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(FastActorCritic, self).__init__()
        # [수정] 학습 코드(30번)와 동일하게 레이어 확장 (256 -> 256 -> 128)
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),  # 추가된 레이어
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 2.0)
        self.act = nn.Tanh()
        self.apply(init_weights)

    def forward(self, x):
        mean = x.mean(dim=1)
        std = x.std(dim=1) + 1e-6
        drift = x[:, -1, :] - x[:, 0, :]
        feat = torch.cat([mean, std, drift], dim=1)
        return self.act(self.actor(self.shared_net(feat))), self.log_std.exp(), None


class PPOAgent:
    def __init__(self, input_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = FastActorCritic(input_dim, action_dim).to(self.device)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        self.policy.eval()
        with torch.no_grad():
            mean, _, _ = self.policy(state)
        return mean.cpu().numpy()


# ==============================================================================
# 2. Utilities
# ==============================================================================
def enu_to_ll(x, y, start_lat, start_lon):
    R = 6378137.0
    d_lat = (y / R) * (180.0 / np.pi)
    d_lon = (x / (R * np.cos(np.radians(start_lat)))) * (180.0 / np.pi)
    return start_lat + d_lat, start_lon + d_lon


class StrapdownNavigator:
    def __init__(self, start_pose, gravity=9.81):
        self.params = gtsam.PreintegrationParams.MakeSharedU(gravity)
        self.params.setAccelerometerCovariance(np.eye(3) * 1e-4)
        self.params.setGyroscopeCovariance(np.eye(3) * 1e-5)
        self.params.setIntegrationCovariance(np.eye(3) * 1e-5)
        self.bias = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
        self.pim = gtsam.PreintegratedImuMeasurements(self.params, self.bias)
        self.curr_pose = start_pose
        self.curr_vel = np.zeros(3)
        self.poses = [start_pose]

    def integrate(self, acc, gyr, dt):
        self.pim.integrateMeasurement(acc, gyr, dt)

    def predict(self):
        state = gtsam.NavState(self.curr_pose, self.curr_vel)
        pred = self.pim.predict(state, self.bias)
        self.curr_pose = pred.pose()
        self.curr_vel = pred.velocity()
        self.pim.resetIntegration()
        self.poses.append(self.curr_pose)
        return self.curr_pose

    def zero_velocity_update(self):
        self.curr_vel = np.zeros(3)


# ==============================================================================
# 3. Main Verification Logic
# ==============================================================================
def run_verification():
    print(">>> [Verification] Loading Best Model...")
    agent = PPOAgent(input_dim=21, action_dim=18)
    model_path = "output_high_end/rl_parallel.pth"
    if os.path.exists(model_path):
        agent.load(model_path)
    else:
        print(f"Model not found at {model_path}")
        return

    # Start Location (Busan)
    START_LOC = (35.1796, 129.0756)
    road_gen = RoadTrajectoryGenerator(START_LOC, 5000)
    sim = TrajectorySimulator(road_gen, 0.1)

    print(">>> Generating 10-minute Test Trajectory...")
    traj = sim.generate_3d_profile(total_duration_min=10)

    nav_raw = StrapdownNavigator(traj[0]["pose"])
    nav_rl = StrapdownNavigator(traj[0]["pose"])
    nav_raw.curr_vel = nav_rl.curr_vel = traj[0]["vel_world"]
    imu = ImuSensor(accel_noise=1e-4, gyro_noise=1e-5)

    # Physics Settings (Match training)
    physics = {
        "acc_h_tanh": 0.005,
        "gyr_h_tanh": 0.0005,
        "coeffs": {
            "acc_b_lin": np.array([0.005] * 3),
            "gyr_b_lin": np.array([5e-5] * 3),
            "acc_s_lin": np.array([0.002] * 3),
            "gyr_s_lin": np.array([0.0005] * 3),
        },
    }
    coeffs = physics["coeffs"]

    obs_buf = []
    path_gt, path_raw, path_rl = [], [], []
    curr_params = {
        "acc_b": np.zeros(3),
        "gyr_b": np.zeros(3),
        "acc_s": np.ones(3),
        "gyr_s": np.ones(3),
        "acc_h": np.zeros(3),
        "gyr_h": np.zeros(3),
    }

    # Logging
    log_keys = ["acc_b", "acc_s", "acc_h", "gyr_b", "gyr_s", "gyr_h"]
    logs = {k: {"true": [], "est": []} for k in log_keys}
    prev_sf = np.zeros(3)
    prev_wb = np.zeros(3)

    print(">>> Running Navigation...")
    for i, d in enumerate(traj):
        gt_p = d["pose"].translation()
        path_gt.append([gt_p.x(), gt_p.y(), gt_p.z()] if hasattr(gt_p, "x") else gt_p)

        ma, mg, _ = imu.measure(d["pose"], d["sf_true"], d["omega_body"], d["temp"])
        dt_t = d["temp"] - 20.0

        # True Physics
        diff_sf = d["sf_true"] - prev_sf
        diff_wb = d["omega_body"] - prev_wb
        t_p = {
            "acc_b": coeffs["acc_b_lin"] * dt_t,
            "acc_s": 1.0 + coeffs["acc_s_lin"] * dt_t,
            "gyr_b": coeffs["gyr_b_lin"] * dt_t,
            "gyr_s": 1.0 + coeffs["gyr_s_lin"] * dt_t,
            "acc_h": physics["acc_h_tanh"] * np.tanh(diff_sf * 50.0),
            "gyr_h": physics["gyr_h_tanh"] * np.tanh(diff_wb * 50.0),
        }
        prev_sf = d["sf_true"]
        prev_wb = d["omega_body"]

        ma = ma * t_p["acc_s"] + t_p["acc_b"] + t_p["acc_h"]
        mg = mg * t_p["gyr_s"] + t_p["gyr_b"] + t_p["gyr_h"]

        if d["speed"] < 0.05:
            nav_raw.zero_velocity_update()
            nav_rl.zero_velocity_update()

        nav_raw.integrate(ma, mg, 0.1)
        nav_raw.predict()
        pr = nav_raw.poses[-1].translation()
        path_raw.append([pr.x(), pr.y(), pr.z()] if hasattr(pr, "x") else pr)

        obs_buf.append(np.concatenate([ma / 9.81, mg, [(d["temp"] - 20) / 30]]))
        if len(obs_buf) > 600:
            obs_buf.pop(0)

        if len(obs_buf) == 600 and i % 10 == 0:
            state_tensor = np.expand_dims(np.array(obs_buf, dtype=np.float32), axis=0)
            a = agent.select_action(state_tensor)[0]

            raw_p = {
                "acc_b": a[0:3] * 0.2,
                "gyr_b": a[3:6] * 0.01,
                "acc_s": a[6:9] * 0.1 + 1.0,
                "gyr_s": a[9:12] * 0.02 + 1.0,
                "acc_h": a[12:15] * 0.01,
                "gyr_h": a[15:18] * 0.001,
            }
            alpha = 0.2
            for k in curr_params:
                curr_params[k] = (1 - alpha) * curr_params[k] + alpha * raw_p[k]

        for k in log_keys:
            logs[k]["true"].append(t_p[k])
            logs[k]["est"].append(curr_params[k])

        c_acc = (ma - curr_params["acc_b"] - curr_params["acc_h"]) / curr_params["acc_s"]
        c_gyr = (mg - curr_params["gyr_b"] - curr_params["gyr_h"]) / curr_params["gyr_s"]

        nav_rl.integrate(c_acc, c_gyr, 0.1)
        nav_rl.predict()
        pl = nav_rl.poses[-1].translation()
        path_rl.append([pl.x(), pl.y(), pl.z()] if hasattr(pl, "x") else pl)

    path_gt = np.array(path_gt)
    path_raw = np.array(path_raw)
    path_rl = np.array(path_rl)
    err_raw = np.linalg.norm(path_gt - path_raw, axis=1)
    err_rl = np.linalg.norm(path_gt - path_rl, axis=1)

    print(f"\n[Final Results]")
    print(f"  > Mean Error (Raw): {np.mean(err_raw):.2f} m")
    print(f"  > Mean Error (RL) : {np.mean(err_rl):.2f} m")

    # 1. Params Plot
    fig, axs = plt.subplots(6, 3, figsize=(18, 20))
    t = np.arange(len(err_raw)) * 0.1 / 60.0
    axes_name = ["X", "Y", "Z"]
    for r, key in enumerate(log_keys):
        true_val = np.array(logs[key]["true"])
        est_val = np.array(logs[key]["est"])
        for c in range(3):
            axs[r, c].plot(t, true_val[:, c], "k--", label="True")
            axs[r, c].plot(t, est_val[:, c], "r-", label="Est")
            axs[r, c].set_title(f"{key} {axes_name[c]}")
            axs[r, c].grid(True)
            if r == 0 and c == 0:
                axs[r, c].legend()
    plt.tight_layout()
    if not os.path.exists("output_verification"):
        os.makedirs("output_verification")
    plt.savefig("output_verification/full_param_analysis.png")
    print("Graph 1 Saved: full_param_analysis.png")

    # 2. Nav Plot
    fig2, axs2 = plt.subplots(1, 2, figsize=(14, 6))
    axs2[0].plot(path_gt[:, 0], path_gt[:, 1], "k-", label="Ground Truth", linewidth=2)
    axs2[0].plot(path_raw[:, 0], path_raw[:, 1], "r--", label="Raw", alpha=0.7)
    axs2[0].plot(path_rl[:, 0], path_rl[:, 1], "b-", label="RL", alpha=0.8)
    axs2[0].set_title("2D Trajectory")
    axs2[0].legend()
    axs2[0].grid(True)
    axs2[0].axis("equal")

    axs2[1].plot(t, err_raw, "r--", label="Raw Error")
    axs2[1].plot(t, err_rl, "b-", label="RL Error")
    axs2[1].set_title("Position Error")
    axs2[1].legend()
    axs2[1].grid(True)
    plt.tight_layout()
    plt.savefig("output_verification/nav_performance.png")
    print("Graph 2 Saved: nav_performance.png")

    # 3. Folium Map
    print(">>> Generating Interactive Map (Folium)...")
    m = folium.Map(location=START_LOC, zoom_start=14)
    ll_gt = [enu_to_ll(p[0], p[1], START_LOC[0], START_LOC[1]) for p in path_gt]
    ll_raw = [enu_to_ll(p[0], p[1], START_LOC[0], START_LOC[1]) for p in path_raw]
    ll_rl = [enu_to_ll(p[0], p[1], START_LOC[0], START_LOC[1]) for p in path_rl]

    folium.PolyLine(ll_gt, color="green", weight=5, opacity=0.6, tooltip="Ground Truth").add_to(m)
    folium.PolyLine(ll_raw, color="red", weight=3, opacity=0.6, tooltip="Raw").add_to(m)
    folium.PolyLine(ll_rl, color="blue", weight=3, opacity=0.8, tooltip="RL Calibrated").add_to(m)

    map_path = "output_verification/trajectory_map.html"
    m.save(map_path)
    print(f"Map Saved: {map_path}")


if __name__ == "__main__":
    run_verification()
