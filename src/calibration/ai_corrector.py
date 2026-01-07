import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class CalibrationLayer(nn.Module):
    """
    물리적 교정 모델을 모사한 신경망 레이어
    Calibrated = W * (Raw - Bias)
    - Bias: 학습 가능한 파라미터 (초기값 0)
    - W: 학습 가능한 파라미터 (초기값 Identity, Scale+Misalignment의 역행렬 역할)
    """

    def __init__(self):
        super(CalibrationLayer, self).__init__()
        # Accel Parameters
        self.accel_bias = nn.Parameter(torch.zeros(3))
        self.accel_W = nn.Parameter(torch.eye(3))  # Inverse of T_accel

        # Gyro Parameters
        self.gyro_bias = nn.Parameter(torch.zeros(3))
        self.gyro_W = nn.Parameter(torch.eye(3))  # Inverse of T_gyro

    def forward(self, raw_accel, raw_gyro):
        # raw shape: (Batch, 3)

        # 1. Remove Bias
        acc_unbiased = raw_accel - self.accel_bias
        gyr_unbiased = raw_gyro - self.gyro_bias

        # 2. Apply Correction Matrix (Scale & Misalignment Correction)
        # y = x @ W.T (Linear layer operation)
        acc_calib = torch.matmul(acc_unbiased, self.accel_W.t())
        gyr_calib = torch.matmul(gyr_unbiased, self.gyro_W.t())

        return acc_calib, gyr_calib


class AiCalibrator:
    def __init__(self):
        self.model = CalibrationLayer()
        # 정밀한 파라미터 튜닝을 위해 학습률 조정
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        self.criterion = nn.MSELoss()

    def train_simulation(self, true_measurements, raw_measurements, epochs=1000):
        """
        :param true_measurements: List of (true_accel, true_gyro) - Ground Truth
        :param raw_measurements: List of (meas_accel, meas_gyro) - Sensor Output
        """
        self.model.train()

        # 데이터 변환 (List -> Tensor)
        # 시계열 전체 데이터를 배치로 사용
        raw_acc = torch.FloatTensor(np.array([m[0] for m in raw_measurements]))
        raw_gyr = torch.FloatTensor(np.array([m[1] for m in raw_measurements]))

        true_acc = torch.FloatTensor(np.array([m[0] for m in true_measurements]))
        true_gyr = torch.FloatTensor(np.array([m[1] for m in true_measurements]))

        print(f"Training AI Corrector (Bias + Scale + Misalignment) for {epochs} epochs...")

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # Forward
            pred_acc, pred_gyr = self.model(raw_acc, raw_gyr)

            # Loss 계산 (Accel과 Gyro 오차 합)
            loss = self.criterion(pred_acc, true_acc) + self.criterion(pred_gyr, true_gyr)

            # Backward
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f"  Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")

    def get_calibration_params(self):
        """학습된 파라미터 추출 (Numpy 포맷)"""
        self.model.eval()
        params = {
            "accel_bias": self.model.accel_bias.detach().numpy(),
            "accel_matrix_inv": self.model.accel_W.detach().numpy(),
            "gyro_bias": self.model.gyro_bias.detach().numpy(),
            "gyro_matrix_inv": self.model.gyro_W.detach().numpy(),
        }
        return params

    def correct(self, raw_measurements):
        """학습된 모델을 사용하여 전체 데이터 보정"""
        self.model.eval()
        with torch.no_grad():
            raw_acc = torch.FloatTensor(np.array([m[0] for m in raw_measurements]))
            raw_gyr = torch.FloatTensor(np.array([m[1] for m in raw_measurements]))

            calib_acc, calib_gyr = self.model(raw_acc, raw_gyr)

        # 결과를 다시 List of tuples 형태로 변환
        corrected_data = []
        c_acc = calib_acc.numpy()
        c_gyr = calib_gyr.numpy()
        for i in range(len(c_acc)):
            corrected_data.append((c_acc[i], c_gyr[i]))

        return corrected_data
