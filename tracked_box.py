import numpy as np
from scipy.linalg import block_diag
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from tracked_point import TrackedPoint


def box_kalman_filter(initial_state, Q_std, R_std):
    """
    Tracks a 2D rectangular object (e.g. a bounding box) whose state includes
    position, velocity, and dimensions.

    Parameters
    ----------
    initial_state : sequence of floats
        [x, vx, y, vy, w, h]
    Q_std : float
        Standard deviation to use for process noise covariance matrix
    R_std : float
        Standard deviation to use for measurement noise covariance matrix

    Returns
    -------
    kf : filterpy.kalman.KalmanFilter instance
    """
    kf = KalmanFilter(dim_x=6, dim_z=4)
    dt = 1.0   # time step

    # state mean and covariance
    kf.x = np.array([initial_state]).T
    kf.P = np.eye(kf.dim_x) * 500.

    # no control inputs
    kf.u = 0.

    # state transition matrix
    kf.F = np.eye(kf.dim_x)
    kf.F[0, 1] = kf.F[2, 3] = dt

    # measurement matrix - maps from state space to observation space, so
    # shape is dim_z x dim_x. Set coefficients for x,y,w,h to 1.0.
    kf.H = np.zeros([kf.dim_z, kf.dim_x])
    kf.H[0, 0] = kf.H[1, 2] = kf.H[2, 4] = kf.H[3, 5] = 1.0

    # measurement noise covariance
    kf.R = np.eye(kf.dim_z) * R_std**2

    # process noise covariance for x-vx or y-vy pairs
    q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_std**2)

    # assume width and height are uncorrelated
    q_wh = np.diag([Q_std**2, Q_std**2])

    kf.Q = block_diag(q, q, q_wh)

    return kf


class TrackedBox(TrackedPoint):
    """Class for tracking a 2D rectangular object (e.g. a bounding box) whose
    state includes position, velocity, width and height. The box is centered
    on (x,y). Extends TrackedPoint."""

    def __init__(self, state, sigma_Q, sigma_R):
        x, vx, y, vy, w, h = state
        TrackedPoint.__init__(self, [x, vx, y, vy], sigma_Q, sigma_R)

        # Override the TrackedPoint kalman filter initialization
        self.kf = box_kalman_filter(state, sigma_Q, sigma_R)

        # Observed box width and height
        self.w = w
        self.h = h

    def step(self, z):
        """See docs for TrackedPoint.step()

        Parameters
        ----------
        z : sequence of floats
            Measurement vector (x, y, w, h)
        """
        TrackedPoint.step(self, z)
        self.w = z[2]
        self.h = z[3]

    def kw(self):
        return self.kf.x[4, 0]

    def kh(self):
        return self.kf.x[5, 0]

    def top_left(self):
        """NOTE: assumes y increases downwards according to image or SVG
        conventions. Coords are rounded to ints"""
        return (int(self.x - self.w/2),
                int(self.y - self.h/2))

    def bottom_right(self):
        """NOTE: assumes y increases downwards according to image or SVG
        conventions. Coords are rounded to ints"""
        return (int(self.x + self.w/2),
                int(self.y + self.h/2))

    def k_top_left(self):
        """NOTE: assumes y increases downwards according to image or SVG
        conventions. Coords are rounded to ints"""
        return (int(self.kx() - self.kw()/2),
                int(self.ky() - self.kh()/2))

    def k_bottom_right(self):
        """NOTE: assumes y increases downwards according to image or SVG
        conventions. Coords are rounded to ints"""
        return (int(self.kx() + self.kw()/2),
                int(self.ky() + self.kh()/2))
