import folium
import numpy as np


class RealMapVisualizer:
    def __init__(self, origin_lat=37.5665, origin_lon=126.9780):
        """
        :param origin_lat: 기준점 위도 (기본값: 서울 시청)
        :param origin_lon: 기준점 경도
        """
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        # 위도 1도당 약 111km, 경도 1도당 약 111km * cos(lat)
        self.m_per_deg_lat = 111000.0
        self.m_per_deg_lon = 111000.0 * np.cos(np.radians(origin_lat))

    def local_to_latlon(self, x, y):
        """Local XY(m) -> Global LatLon"""
        d_lat = y / self.m_per_deg_lat
        d_lon = x / self.m_per_deg_lon
        return self.origin_lat + d_lat, self.origin_lon + d_lon

    def save_map(self, poses, filename="trajectory_map.html"):
        """GTSAM Pose 리스트를 받아 지도 HTML 파일 생성"""
        # 중심점 계산을 위해 경로 변환
        path_coords = []
        for p in poses:
            lat, lon = self.local_to_latlon(p.x(), p.y())
            path_coords.append([lat, lon])

        if not path_coords:
            print("No poses to plot on map.")
            return

        # 지도 생성 (시작점 기준)
        m = folium.Map(location=path_coords[0], zoom_start=15)

        # 궤적 그리기 (PolyLine)
        folium.PolyLine(
            locations=path_coords,
            color="blue",
            weight=5,
            opacity=0.7,
            tooltip="Estimated Trajectory",
        ).add_to(m)

        # 시작/끝 마커
        folium.Marker(path_coords[0], popup="Start", icon=folium.Icon(color="green")).add_to(m)
        folium.Marker(path_coords[-1], popup="End", icon=folium.Icon(color="red")).add_to(m)

        m.save(filename)
        print(f"Map saved to '{filename}'. Open this file in your browser.")
