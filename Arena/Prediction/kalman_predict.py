import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
from Prediction.predictor import TrajectoryPredictor


def create_kalman_filter(
    init_x=None, input_dim=2, num_terms=2, dt=1, r_var=5.0, q_var=0.1
):
    """
    Return a constant velocity kalman filter
    input_dim - dimensions of input
    num_derivatives - number of terms in the dynamic system (i.e. 2 for constant velocity, 3 for constant accel, etc.).
    dt - time step
    r_var - variance of state uncertainty matrix R
    q_var - variance of white noise matrix Q
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

    # Measurement function
    h_block = np.zeros(num_terms, dtype=np.float)
    h_block[0] = 1.0
    kf.H = block_diag(*([h_block] * input_dim))

    kf.P *= 1.0  # covariance matrix
    kf.R *= r_var  # state uncertainty

    q = Q_discrete_white_noise(dim=num_terms, dt=dt, var=q_var)
    kf.Q = block_diag(*([q] * input_dim))

    return kf


class KalmanPredictor(TrajectoryPredictor):
    def __init__(self, forecast_horizon, num_derivatives, q_var=0.01, r_var=5.0):
        super().__init__(forecast_horizon)
        self.num_terms = num_derivatives + 1
        self.q_var = q_var
        self.r_var = r_var

    def init_trajectory(self, detection):
        init_x1 = np.zeros(2 * self.num_terms, np.float)
        init_x1[0 :: self.num_terms] = detection[:2]

        init_x2 = np.zeros(2 * self.num_terms, np.float)
        init_x2[0 :: self.num_terms] = detection[2:4]

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

    def update_and_predict(self, history):
        self.kf1.predict()
        self.kf2.predict()

        if not np.isnan(history[-1, 0]):
            self.kf1.update(history[-1, :2])
            self.kf2.update(history[-1, 2:])

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
