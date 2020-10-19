"""
Module for trajectory prediction using Kalman filters.

create_kalman_filter - a function for creating a Kalman filter object based on the filterpy library
KalmanPredictor - a subclass of TrajectoryPredictor that generates trajectory forecasts using kalman filters.
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
from Prediction.predictor import TrajectoryPredictor


def create_kalman_filter(
    init_x=None, input_dim=2, num_terms=2, dt=1, r_var=5.0, q_var=0.1
):
    """
    Create a Kalman filter object with the specified number of dimensions and derivatives.
    Input dimensions are assumed to be independent random variables with equal noise
    variance for each dimension.

    :param init_x: initial state vector (size: input_dim * num_terms)
    :param input_dim: dimensions of input
    :param num_terms: number of terms in the dynamic system (i.e. 2 for constant velocity, 3 for constant accel, etc.).
    (defualt: 2, i.e, constant velocity)
    :param dt: time step
    :param r_var: variance of measurement noise matrix R
    :param q_var: variance of process white noise matrix Q

    :return: Kalman filter object
    """

    kf = KalmanFilter(dim_x=input_dim * num_terms, dim_z=input_dim)
    if init_x is None:
        kf.x = np.zeros(input_dim * num_terms)
    else:
        kf.x = init_x

    # state transition matrix
    f = np.zeros((num_terms, num_terms), dtype=np.float)
    for i in range(num_terms):
        for j in range(i, num_terms):
            p = j - i
            f[i, j] = (dt ** p) / max(1, p)

    kf.F = block_diag(*([f] * input_dim))

    # measurement function
    h_block = np.zeros(num_terms, dtype=np.float)
    h_block[0] = 1.0
    kf.H = block_diag(*([h_block] * input_dim))

    kf.P *= 1.0  # covariance matrix
    kf.R *= r_var  # measurement noise variance

    # process noise covariance matrix
    q = Q_discrete_white_noise(dim=num_terms, dt=dt, var=q_var)
    kf.Q = block_diag(*([q] * input_dim))

    return kf


class KalmanPredictor(TrajectoryPredictor):
    """
    A TrajectoryPredictor that uses a Kalman filter to predict future coordinates.

    The predictor uses two kalman filters, one for each corner of the detected bounding box. This prevents dependence
    between the state coordinates of the filter.

    Predictions are made by recursively applying the state transition matrix F on the state vector x for each time step.
    That is, the prediction at time t0 for time step t0 + t is: (F ** t) * x

    """

    def __init__(self, forecast_horizon, num_derivatives, q_var=0.01, r_var=5.0):
        """
        Initialize a KalmanPredictor.

        :param forecast_horizon: number of iterations to project the current state vector ahead with the transition matrix
        :param num_derivatives: degree of the system (constant velocity - 1, constant acceleration - 2, ..etc).
        :param q_var: variance of process white noise matrix Q
        :param r_var: variance of measurement noise matrix R
        """
        super().__init__(1, forecast_horizon)
        self.num_terms = num_derivatives + 1
        self.q_var = q_var
        self.r_var = r_var

        self.discontinuity = False

    def init_trajectory(self, detection):
        """
        Initialize the kalman filters with the supplied detection for the first (position) term
        and 0 for all other terms (such as velocity and acceleration).

        This is also used after a sequence of NaN detections (implying a discontinuity in the trajectory).

        :param detection: The initial bbox detection.
        """
        self.discontinuity = False

        init_x1 = np.zeros(2 * self.num_terms, np.float)
        init_x1[0 :: self.num_terms] = detection[:2]  # x1y1

        init_x2 = np.zeros(2 * self.num_terms, np.float)
        init_x2[0 :: self.num_terms] = detection[2:4]  # x2y2

        self.kf1 = create_kalman_filter(
            init_x=init_x1,
            input_dim=2,
            num_terms=self.num_terms,
            q_var=self.q_var,
            r_var=self.r_var,
        )

        self.kf2 = create_kalman_filter(
            init_x=init_x2,
            input_dim=2,
            num_terms=self.num_terms,
            q_var=self.q_var,
            r_var=self.r_var,
        )

    def _update_and_predict(self, past_input):
        """
        Update the filter, calculate a new state vector according to the last value in past_input, and
        generate a trajectory forecast.
        Initialize the filters when the first non NaN value arrives, and reinitializes after a sequence of NaNs.

        :param past_input: A numpy array of past detections. Only the last detection is used.
        :return: A forecast numpy array of shape (forecast_horizon, 4) containing forecasts for the next timesteps.
        """
        if not np.isnan(past_input[-1, 0]):
            if self.discontinuity:
                # Reinitialize trajectory after a discontinuity (a sequence of NaNs has ended)
                self.init_trajectory(past_input[-1])
            else:
                # Update the filters according to the new measurement
                self.kf1.predict()
                self.kf2.predict()

                self.kf1.update(past_input[-1, :2])
                self.kf2.update(past_input[-1, 2:])
        else:
            self.discontinuity = True

        # Generate a trajectory forecast
        forecast = np.empty((self.forecast_horizon, 4), np.float)
        pred1 = self.kf1.x
        pred2 = self.kf2.x

        for i in range(self.forecast_horizon):
            forecast[i] = np.concatenate(
                (pred1[0 :: self.num_terms], pred2[0 :: self.num_terms])
            )
            pred1 = np.dot(self.kf1.F, pred1)
            pred2 = np.dot(self.kf2.F, pred2)

        return forecast
