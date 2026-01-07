import osmnx as ox
import networkx as nx
import numpy as np
import gtsam
from scipy.interpolate import CubicSpline


class RoadTrajectoryGenerator:
    def __init__(self, location_point=(35.1796, 129.0756), dist=1000):
        """
        :param location_point: 지도 중심 (Lat, Lon) - 예: 부산 시청
        :param dist: 반경 (m)
        """
        print(f"Downloading road network around {location_point}...")
        # 도로망 다운로드 (Drive 타입) -> self.G (원본, 위경도 포함)
        self.G = ox.graph_from_point(location_point, dist=dist, network_type="drive")
        # UTM 좌표계(미터 단위)로 투영 -> self.G_proj (미터 단위 x, y)
        self.G_proj = ox.project_graph(self.G)
        print("Road network loaded.")

    def generate_path(self, start_node=None, end_node=None, num_waypoints=None):
        """임의의(또는 지정된) 시작/끝 노드를 연결하는 경로 생성"""
        nodes = list(self.G_proj.nodes)

        if start_node is None:
            start_node = np.random.choice(nodes)
        if end_node is None:
            end_node = np.random.choice(nodes)

        # 최단 경로 탐색 (노드 ID 리스트 반환)
        try:
            route_ids = nx.shortest_path(self.G_proj, start_node, end_node, weight="length")
        except nx.NetworkXNoPath:
            print("No path found. Trying again with random nodes.")
            return self.generate_path(num_waypoints=num_waypoints)

        # 경로가 너무 짧으면 다시 생성
        if len(route_ids) < 5:
            return self.generate_path(num_waypoints=num_waypoints)

        # 노드 좌표 추출 (x: meter, y: meter) - G_proj 사용
        x_coords = [self.G_proj.nodes[n]["x"] for n in route_ids]
        y_coords = [self.G_proj.nodes[n]["y"] for n in route_ids]

        # 시작점을 (0,0)으로 이동 (Local Frame 원점 설정)
        self.origin_x = x_coords[0]
        self.origin_y = y_coords[0]

        x_local = np.array(x_coords) - self.origin_x
        y_local = np.array(y_coords) - self.origin_y

        return x_local, y_local, route_ids

    def interpolate_trajectory(self, x_points, y_points, target_speed=10.0, dt=0.1):
        """
        불연속적인 경로 점들을 부드러운 궤적(Pose, Vel, Accel)으로 변환
        :param target_speed: 목표 이동 속도 (m/s)
        :param dt: 시뮬레이션 시간 간격
        """
        # 1. 누적 거리 계산 (Path Parameter s)
        diffs = np.sqrt(np.diff(x_points) ** 2 + np.diff(y_points) ** 2)
        dists = np.cumsum(diffs)
        dists = np.insert(dists, 0, 0)  # 시작점 거리 0
        total_dist = dists[-1]

        # 2. 시간 축 생성 (등속 운동 가정)
        total_time = total_dist / target_speed
        t_original = dists / target_speed
        t_new = np.arange(0, total_time, dt)

        # 3. Cubic Spline 보간 (부드러운 곡선 생성)
        # 시간 t에 대한 x(t), y(t) 함수 생성
        cs_x = CubicSpline(t_original, x_points)
        cs_y = CubicSpline(t_original, y_points)

        # 4. 물리량 계산
        # 위치
        pos_x = cs_x(t_new)
        pos_y = cs_y(t_new)
        pos_z = np.zeros_like(pos_x)  # 평지 가정

        # 속도 (1차 미분)
        vel_x = cs_x(t_new, 1)
        vel_y = cs_y(t_new, 1)

        # 가속도 (2차 미분)
        acc_x = cs_x(t_new, 2)
        acc_y = cs_y(t_new, 2)

        # Heading (Yaw) 계산
        yaw = np.arctan2(vel_y, vel_x)

        # 각속도 (Yaw 미분) - 불연속점 처리를 위해 np.gradient 사용
        yaw_unwrap = np.unwrap(yaw)
        omega_z = np.gradient(yaw_unwrap, dt)

        # 데이터 패키징
        trajectory_data = []

        for i in range(len(t_new)):
            # Pose3
            rot = gtsam.Rot3.Ypr(yaw[i], 0, 0)
            point = gtsam.Point3(pos_x[i], pos_y[i], pos_z[i])
            pose = gtsam.Pose3(rot, point)

            # Kinematics
            # World Frame Velocities/Accels
            v_world = np.array([vel_x[i], vel_y[i], 0])
            a_world = np.array([acc_x[i], acc_y[i], 0])

            # Body Frame 변환 (IMU 입력용)
            R = rot.matrix()
            v_body = R.T @ v_world
            a_body = R.T @ a_world
            omega_body = R.T @ np.array([0, 0, omega_z[i]])

            trajectory_data.append(
                {
                    "time": t_new[i],
                    "pose": pose,
                    "accel_body": a_body,  # Kinematic Accel (without gravity)
                    "omega_body": omega_body,
                    "vel_body": v_body,
                }
            )

        return trajectory_data

    def get_latlon_route(self, route_ids):
        """시각화를 위해 경로 노드들의 원본 Lat/Lon 반환"""
        # [수정] G_proj['lat'] 대신 원본 G의 ['y'](Lat), ['x'](Lon) 사용
        lats = [self.G.nodes[n]["y"] for n in route_ids]
        lons = [self.G.nodes[n]["x"] for n in route_ids]
        return list(zip(lats, lons))
