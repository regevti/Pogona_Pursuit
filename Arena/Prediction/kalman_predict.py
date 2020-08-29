from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
import numpy as np
import pandas as pd

from Prediction.predictor import TrajectoryPredictor


def constant_velocity_kalman_filter(init_x=None, dt=1, r_var=5.0, q_var=0.1, dim=2):
    """
    Return a constant velocity kalman filter
    dt - time step
    r_var - variance of state uncertainty matrix R
    q_var - variance of white noise matrix Q
    dim - dimensions of input
    """

    kf = KalmanFilter(dim_x=dim * 2, dim_z=dim)
    if init_x is None:
        kf.x = np.zeros(dim * 2)
    else:
        kf.x = init_x

    # state transition matrix
    f = np.array([[1.0, dt], [0.0, 1.0]])
    kf.F = block_diag(*([f] * dim))

    # Measurement function
    kf.H = block_diag(*([np.array([[1.0, 0.0]])] * dim))

    kf.P *= 1.0  # covariance matrix
    kf.R *= r_var  # state uncertainty

    q = Q_discrete_white_noise(dim=dim, dt=dt, var=q_var)
    kf.Q = block_diag(q, q)

    return kf


class ConstantVelocityKalmanPredictor(TrajectoryPredictor):
    def init_trajectory(self, detection):
        init_x = np.zeros(8, np.float)
        init_x[0::2] = detection[:4]

        self.kf = constant_velocity_kalman_filter(init_x=init_x, dim=4)

    def update_and_predict(self, history):
        self.kf.predict()
        if not np.isnan(history[-1, 0]):
            self.kf.update(history[-1])

        forecast = np.empty((self.forecast_horizon, 4), np.float)
        pred = self.kf.x

        for i in range(self.forecast_horizon):
            forecast[i] = pred[0::2]
            pred = np.dot(self.kf.F, pred)

        return forecast
