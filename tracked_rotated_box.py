import numpy as np
from scipy.linalg import block_diag
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from tracked_box import TrackedBox


def rot_box_kalman_filter(initial_state, Q_std, R_std):
    """
    Tracks a 2D rectangular object (e.g. a bounding box) whose state includes
    position, centroid velocity, dimensions, and rotation angle.

    Parameters
    ----------
    initial_state : sequence of floats
        [x, vx, y, vy, w, h, phi]
    Q_std : float
        Standard deviation to use for process noise covariance matrix
    R_std : float
        Standard deviation to use for measurement noise covariance matrix

    Returns
    -------
    kf : filterpy.kalman.KalmanFilter instance
    """
    kf = KalmanFilter(dim_x=7, dim_z=5)
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
    # shape is dim_z x dim_x.
    kf.H = np.zeros([kf.dim_z, kf.dim_x])

    # z = Hx. H has nonzero coefficients for the following components of kf.x:
    #   x            y            w            h           phi
    kf.H[0, 0] = kf.H[1, 2] = kf.H[2, 4] = kf.H[3, 5] = kf.H[4, 6] = 1.0

    # measurement noise covariance
    kf.R = np.eye(kf.dim_z) * R_std**2

    # process noise covariance for x-vx or y-vy pairs
    q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_std**2)

    # diagonal process noise sub-matrix for width, height, and phi
    qq = Q_std**2*np.eye(3)

    # process noise covariance matrix for full state
    kf.Q = block_diag(q, q, qq)

    return kf


def wrap2pi(a):
    if 0 <= a <= 2*np.pi:
        return a
    if a < 0:
        return a + 2*np.pi
    if a > 2*np.pi:
        return a - 2*np.pi


class TrackedRotatedBox(TrackedBox):
    """Class for tracking a 2D rectangular object (e.g. a bounding box) whose
    rotation is a dynamical variable. The box is centered
    on (x,y). Extends TrackedPoint."""

    def __init__(self, state, sigma_Q, sigma_R):
        x, vx, y, vy, w, h, phi = state
        TrackedBox.__init__(self, state[:-1], sigma_Q, sigma_R)

        # Override the TrackedBox kalman filter initialization
        self.kf = rot_box_kalman_filter(state, sigma_Q, sigma_R)

        # Observed box width and height
        self.phi = phi

    def step(self, z):
        """See docs for TrackedPoint.step()

        Parameters
        ----------
        z : sequence of floats
            Measurement vector (x, y, w, h, phi)
        """

        assert len(z) == 5, \
            'Expected z of length 5 [x,y,w,h,phi], received {}'.format(z)

        TrackedBox.step(self, z)
        # self.phi = wrap2pi(z[4])
        self.phi = z[4]

    def kphi(self):
        # return wrap2pi(self.kf.x[6, 0])
        return self.kf.x[6, 0]
