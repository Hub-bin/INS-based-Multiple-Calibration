# INS-based Multiple Sensor Calibration Simulator

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![GTSAM](https://img.shields.io/badge/GTSAM-4.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)

**INS-based Multiple Sensor Calibration Simulator**는 로보틱스 및 자율주행 시스템을 위한 다중 센서 캘리브레이션(Multi-Sensor Calibration) 알고리즘을 개발하고 검증하기 위한 Python 기반 시뮬레이션 프레임워크입니다.

[Georgia Tech Smoothing and Mapping (GTSAM)](https://gtsam.org/) 라이브러리를 기반으로 구축되었으며, IMU, Camera, LiDAR 등 다양한 센서의 데이터를 생성하고, Factor Graph 최적화를 통해 센서의 내/외부 파라미터(Intrinsics/Extrinsics) 및 바이어스(Bias)를 추정하는 기능을 제공합니다.

---

## 🚀 Key Features

### 1. High-Fidelity Sensor Simulation
다양한 노이즈 모델과 물리적 특성을 반영한 센서 데이터를 생성합니다.
* **IMU (6-DOF)**: 가속도계(Accelerometer) 및 자이로스코프(Gyroscope)의 White Noise, Random Walk Bias, 중력 가속도, 코리올리 힘 등을 시뮬레이션.
* **Camera (Pinhole Model)**: 3D 랜드마크의 2D 투영(Projection), 렌즈 왜곡(Distortion), 이미지 노이즈, FOV(Field of View) 필터링 지원.
* **LiDAR (3D Point Cloud)**: 거리(Range) 및 수직/수평 시야각(FOV) 제한, 거리/각도 측정 노이즈 반영.

### 2. Advanced Calibration Algorithms
GTSAM의 Factor Graph를 활용한 최신 캘리브레이션 알고리즘을 구현했습니다.
* **Offline Calibration (Batch)**: `Levenberg-Marquardt Optimizer`를 사용하여 전체 데이터를 일괄 최적화, 정밀한 IMU Bias 추정.
* **Online Calibration (Incremental)**: `iSAM2` 알고리즘을 적용하여 실시간으로 들어오는 데이터에 대해 센서 파라미터를 점진적으로 갱신 및 추정.
* **Extrinsic Calibration (LiDAR-Camera)**: 3D-2D 매칭 쌍(Correspondences)을 이용한 센서 간 상대 위치(Rigid Body Transform) 최적화.
* **Hand-Eye Calibration (IMU-Camera)**: 이동하는 차량의 Body 궤적과 Visual Odometry 궤적을 비교하여 센서 간의 기하학적 관계($T_{body}^{cam}$) 추정.

### 3. Dynamics Modeling
* **Ground Vehicle**: Ackermann 조향 모델 등을 기반으로 한 지상 이동체 운동학 모델링.

---

## 📂 Project Structure

```bash
INS-based-Multiple-Calibration/
├── examples/                   # 사용 예제 및 시뮬레이션 스크립트
│   ├── 01_imu_simulation.py
│   ├── 02_camera_view.py
│   ├── 03_lidar_scan.py
│   ├── 04_online_calibration.py
│   └── 05_lidar_camera_calibration.py
├── src/                        # 핵심 소스 코드
│   ├── calibration/            # 캘리브레이션 알고리즘 (Offline, Online, Extrinsics, Hand-Eye)
│   ├── dynamics/               # 이동체 운동 모델 (Ground)
│   └── sensors/                # 센서 모델 (IMU, Camera, LiDAR)
├── main.py                     # 메인 실행 파일 (현재: Hand-Eye Calibration 데모)
├── pyproject.toml              # Ruff 설정 파일
└── README.md
```

---

## 🛠️ Installation

본 프로젝트는 Python 3.8 이상 환경을 권장합니다.

1.  **Repository Clone**
    ```bash
    git clone [https://github.com/Hub-bin/INS-based-Multiple-Calibration.git](https://github.com/Hub-bin/INS-based-Multiple-Calibration.git)
    cd INS-based-Multiple-Calibration
    ```

2.  **Dependencies Installation**
    필수 라이브러리(`gtsam`, `numpy`, `scipy`, `matplotlib`)를 설치합니다.
    ```bash
    pip install gtsam numpy scipy matplotlib
    ```

3.  **Dev Tools (Optional)**
    코드 포맷팅을 위해 `ruff`를 사용합니다.
    ```bash
    pip install ruff
    ```

---

## 💻 Usage & Examples

### 1. Hand-Eye Calibration (IMU-Camera)
S자 주행 궤적을 시뮬레이션하고, IMU(Body)와 Camera 간의 상대 위치를 추정합니다.
```bash
python main.py
```
* **Output**: True Extrinsics vs Optimized Result 비교, Translation/Rotation 오차 출력.

### 2. LiDAR-Camera Extrinsic Calibration
LiDAR의 3D 포인트와 Camera의 2D 이미지 좌표 매칭을 통해 두 센서 간의 변환 행렬을 찾습니다.
```bash
python examples/05_lidar_camera_calibration.py
```

### 3. Online IMU Bias Estimation
iSAM2를 사용하여 실시간으로 IMU의 가속도/자이로 바이어스가 수렴하는 과정을 시각화합니다.
```bash
python examples/04_online_calibration.py
```

### 4. Sensor Simulation Visualization
각 센서의 동작을 시각적으로 확인할 수 있습니다.
```bash
python examples/03_lidar_scan.py   # LiDAR FOV 및 Point Cloud 시각화
python examples/02_camera_view.py  # Camera FOV 및 랜드마크 투영 시각화
```
