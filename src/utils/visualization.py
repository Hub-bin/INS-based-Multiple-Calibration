import matplotlib.pyplot as plt
import folium
import numpy as np
import os


class CalibVisualizer:
    def __init__(self, output_dir="output"):
        self.out_dir = output_dir
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def plot_params(self, log, true_acc, true_gyr, dt):
        times = np.array(log["time"]) / 60.0
        param_groups = [
            ("Bias", "bias", "bias"),
            ("Scale", "scale", "scale"),
            ("TempL", "temp_lin", "temp_lin"),
            ("TempN", "temp_non", "temp_non"),
            ("Hyst", "hyst", "hyst"),
        ]

        for sens_name, true_p in [("acc", true_acc), ("gyr", true_gyr)]:
            fig, axes = plt.subplots(5, 3, figsize=(15, 15))
            fig.suptitle(f"{sens_name.upper()} Parameters", fontsize=16)

            for r, (name, l_key, t_key) in enumerate(param_groups):
                est = np.array(log[sens_name][l_key])
                true = true_p[t_key]
                for c in range(3):
                    ax = axes[r, c]
                    ax.plot(times, est[:, c], "b-")
                    tv = true[c] if hasattr(true, "__len__") else true
                    if name == "Scale" and not hasattr(true, "__len__"):
                        tv = 1.0
                    ax.axhline(tv, color="r", ls="--")
                    ax.set_title(f"{name} {'XYZ'[c]}")
                    ax.grid(True)
            plt.tight_layout()
            plt.savefig(f"{self.out_dir}/{sens_name}_params.png")
            plt.close()

    def plot_nav_error(self, gt_traj, poses_dict, dt):
        # [수정] 함수 이름 통일: get_xy
        def get_xy(plist):
            return np.array(
                [
                    (p.x(), p.y()) if hasattr(p, "x") else (p.translation()[0], p.translation()[1])
                    for p in plist
                ]
            )

        gt_xy = get_xy([d["pose"] for d in gt_traj])
        min_l = len(gt_xy)

        plt.figure(figsize=(10, 5))
        for name, poses in poses_dict.items():
            est_xy = get_xy(poses)
            l = min(len(est_xy), min_l)
            # min_l = l # Don't shrink min_l based on bad estimates, just slice

            # Safe slice
            gt_slice = gt_xy[:l]
            est_slice = est_xy[:l]

            err = np.linalg.norm(gt_slice - est_slice, axis=1)
            t_ax = np.arange(l) * dt / 60.0
            plt.plot(t_ax, err, label=name)

        plt.title("Navigation Position Error")
        plt.xlabel("Time (min)")
        plt.ylabel("Error (m)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.out_dir}/nav_error.png")
        plt.close()

    def save_map(self, gt_traj, poses_dict, start_loc):
        m = folium.Map(location=start_loc, zoom_start=14)
        m_deg = 111000.0
        lon_s = m_deg * np.cos(np.radians(start_loc[0]))

        def to_coords(plist):
            return [
                [
                    start_loc[0] + (p.y() if hasattr(p, "x") else p.translation()[1]) / m_deg,
                    start_loc[1] + (p.x() if hasattr(p, "x") else p.translation()[0]) / lon_s,
                ]
                for p in plist
            ]

        # GT
        gt_coords = to_coords([d["pose"] for d in gt_traj])
        folium.PolyLine(gt_coords[::10], color="green", weight=4, opacity=0.6, tooltip="GT").add_to(
            m
        )

        colors = ["red", "purple", "orange", "blue"]
        for i, (name, poses) in enumerate(poses_dict.items()):
            coords = to_coords(poses)
            c = colors[i % len(colors)]
            folium.PolyLine(coords[::10], color=c, weight=2, tooltip=name).add_to(m)

        m.save(f"{self.out_dir}/final_map.html")
