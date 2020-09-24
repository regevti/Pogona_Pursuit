import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag
from Prediction.predictor import TrajectoryPredictor


def create_kalman_filter(
        init_x=None, input_dim=2, num_terms=2, dt=1, r_var=5.0, q_var=0.1
):
    """
    Create a Kalman filter object with the specified number of dimensions and derivatives
    :param init_x:
    :param input_dim: dimensions of input (defualt: 2)
    :param num_terms: number of terms in the dynamic system (i.e. 2 for constant velocity, 3 for constant accel, etc.).
    (defualt: 2, i.e, constant velocity)
    :param dt: time step (defualt: 1)
    :param r_var: variance of state uncertainty matrix R (default: 5)
    :param q_var: variance of white noise matrix Q (default: 0.1)
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
        """
        A TrajectoryPredictor that use a Kalman filter to predict future coordinates
        :param forecast_horizon: number of iterations to project the current state vector ahead with the transition matrix
        :param num_derivatives: degree of the system (constant velocity - 2, acceleration - 3, ..etc).
        :param q_var: variance of white noise matrix Q (default: 0.1)
        :param r_var: variance of state uncertainty matrix R (default: 5)
        """
        super().__init__(1, forecast_horizon)
        self.num_terms = num_derivatives + 1
        self.q_var = q_var
        self.r_var = r_var

        self.discontinuity = False

    def init_trajectory(self, detection):
        self.discontinuity = False

        init_x1 = np.zeros(2 * self.num_terms, np.float)
        init_x1[0:: self.num_terms] = detection[:2]  # x1y1

        init_x2 = np.zeros(2 * self.num_terms, np.float)
        init_x2[0:: self.num_terms] = detection[2:4]  # x2y2

        self.kf1 = create_kalman_filter(init_x=init_x1,
                                        input_dim=2,
                                        num_terms=self.num_terms,
                                        q_var=self.q_var,
                                        r_var=self.r_var,
                                        )

        self.kf2 = create_kalman_filter(init_x=init_x2,
                                        input_dim=2,
                                        num_terms=self.num_terms,
                                        q_var=self.q_var,
                                        r_var=self.r_var,
                                        )

    def _update_and_predict(self, past_input):
        if not np.isnan(past_input[-1, 0]):
            if self.discontinuity:
                # Reinitialize trajectory after a discontinuity
                # Perhaps this is too extreme?
                self.init_trajectory(past_input[-1])
            else:
                self.kf1.predict()
                self.kf2.predict()

                self.kf1.update(past_input[-1, :2])
                self.kf2.update(past_input[-1, 2:])
        else:
            self.discontinuity = True

        forecast = np.empty((self.forecast_horizon, 4), np.float)
        pred1 = self.kf1.x
        pred2 = self.kf2.x

        for i in range(self.forecast_horizon):
            forecast[i] = np.concatenate((pred1[0:: self.num_terms], pred2[0:: self.num_terms]))
            pred1 = np.dot(self.kf1.F, pred1)
            pred2 = np.dot(self.kf2.F, pred2)

        return forecast
