from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
import numpy as np
import pandas as pd

from Prediction.predictor import TrajectoryPredictor
from Prediction.detector import xywh_to_centroid


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


def batch_predict_hits(f, centroids, y_threshold=930, max_timesteps=60):
    """
    Run the filter in order to predict the number of time steps until passing
    the given y coordinate.

    f - The filter used for prediction. Expecting predict(), update(x)
        functions, and an x property.
    y_threshold - When passing this y value we assume a hit has occurred.
    max_timesteps - maximum number of time steps to predict.

    Return a data frame containing position, velocity, predicted position and
    number of time steps until the predicted hit (or nan if there's no hit in
    max_timesteps steps).
    """
    cents_pred = np.zeros((centroids.shape[0], 7))
    cents_pred[:] = np.nan

    for i in range(len(centroids)):
        meas = centroids[i, :2]
        if np.isnan(meas[0]) or np.isnan(meas[1]):
            continue

        f.predict()
        f.update(meas)

        cents_pred[i, :4] = f.x
        new_pred = f.x

        for j in range(max_timesteps):
            new_pred = np.dot(f.F, new_pred)
            pred_x, pred_y = new_pred[0], new_pred[2]
            if pred_y > y_threshold or j == max_timesteps - 1:
                cents_pred[i, 4:] = np.array([pred_x, pred_y, j])
                break

    cents_df = pd.DataFrame(
        data=cents_pred, columns=["x", "vx", "y", "vy", "pred_x", "pred_y", "k"]
    )
    cents_df = pd.concat(
        [pd.DataFrame(centroids, columns=["det_x", "det_y"]), cents_df], axis=1
    )

    return cents_df


class ConstantVelocityKalmanPredictor(TrajectoryPredictor):
    def init_trajectory(self, detection):
        init_x = np.zeros(4, np.float)
        init_x[0::2] = xywh_to_centroid(detection[:4])

        self.kf = constant_velocity_kalman_filter(init_x=init_x, dim=2)

    def update_and_predict(self, history):
        self.kf.predict()
        if not np.isnan(history[-1, 0]):
            self.kf.update(xywh_to_centroid(history[-1]))

        forecast = np.empty((self.forecast_horizon, 2), np.float)
        pred = self.kf.x

        for i in range(self.forecast_horizon):
            forecast[i] = pred[0::2]
            pred = np.dot(self.kf.F, pred)

        return forecast
