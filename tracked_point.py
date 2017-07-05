from collections import deque
import numpy as np
from numpy.linalg import norm, eig

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

from scipy.linalg import block_diag


def point_2d_kalman_filter(initial_state, Q_std, R_std):
    """
    Parameters
    ----------
    initial_state : sequence of floats
        [x0, vx0, y0, vy0]
    Q_std : float
        Standard deviation to use for process noise covariance matrix
    R_std : float
        Standard deviation to use for measurement noise covariance matrix

    Returns
    -------
    kf : filterpy.kalman.KalmanFilter instance
    """
    kf = KalmanFilter(dim_x=4, dim_z=2)
    dt = 1.0   # time step

    # state mean (x, vx, y, vy) and covariance
    kf.x = np.array([initial_state]).T
    kf.P = np.eye(kf.dim_x) * 500.

    # no control inputs
    kf.u = 0.

    # state transition matrix
    kf.F = np.array([[1, dt, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, dt],
                     [0, 0, 0, 1]])

    # measurement matrix - maps from state space to observation space
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0]])

    # measurement noise covariance
    kf.R = np.eye(kf.dim_z) * R_std**2

    # process noise covariance
    q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_std**2)
    kf.Q = block_diag(q, q)

    return kf


def add_tracked_point(tracked_points, x, y, bounds,
                      sigma_proc, sigma_meas, max_n_coasts=3):
    """Append a new TrackedPoint object to the `tracked_points` list.

    Parameters
    ----------
    tracked_points : list(TrackedPoint)
        Append a new TrackedPoint to this list
    x, y : float
        Initial position
    bounds : sequence of floats
        (xmin, ymin, xmax, ymax)
    sigma_proc, sigma_meas : float
        Process and measurement noise sigma parameters (respectively)
        for new tracks
    max_n_coasts :  int
        Maximum number of frames to drift without a measurement
    """
    xmin, ymin, xmax, ymax = bounds
    tp = TrackedPoint(x, y, 0, 0, sigma_proc, sigma_meas)
    tp.boundary.xmin, tp.boundary.ymin = xmin, ymin
    tp.boundary.xmax, tp.boundary.ymax = xmax, ymax
    tp.max_n_coasts = max_n_coasts
    tp.id = add_tracked_point.id
    add_tracked_point.id += 1
    tracked_points.append(tp)


add_tracked_point.id = 0


def match_tracks_to_observations(tracked_objects,
                                 observations,
                                 bounds,
                                 distance_threshold=30,
                                 sigma_proc=0.1,
                                 sigma_meas=10,
                                 max_n_coasts=3,
                                 min_lifetime=3):
    """Associate tracks to observations. The `tracked_objects` and
    `observations` lists are modified in-place.

    Parameters
    ----------
    tracked_objects : sequence of TrackedPoint instances
    observations : sequence of int or float pairs
        list of observed (x,y) positions
    bounds : sequence of floats
        (xmin, ymin, xmax, ymax)
    distance_threshold : int
        Maximum distance to nearest observation for matching
    sigma_proc, sigma_meas : float, optional
        Process and measurement noise sigma parameters (respectively)
        for new tracks
    max_n_coasts : int, optional, default=3
        Max number of time steps the track can propagate without an observation
        (prediction-only) before being removed

    Returns
    -------
    finished_tracks : list of TrackedPoints
        Good tracks that have run their course
    """
    finished_tracks = []

    # Advance each tracker to the nearest available measurement.
    # If no nearby measurement is found in this frame, coast.
    for track in tracked_objects:
        nearest_observation, distance = track.nearest_observation(observations)
        if distance < distance_threshold:
            track.step_to(nearest_observation)
            observations.remove(nearest_observation)
            track.n_coasts = 0
        else:
            track.coast()

    # Handle lost or out-of-bounds tracks. If the track is disappearing
    # after a good run, transfer it to the `finished_tracks` list.
    for t in tracked_objects:
        if t.is_valid():
            pass
        else:
            if t.lifetime > min_lifetime and t.n_coasts < t.max_n_coasts:
                finished_tracks.append(t)
            tracked_objects.remove(t)

    # Start tracking any remaining measurements under the assumption
    # that they are new (not yet tracked).
    for i, observation in enumerate(observations):
        x, y = observation
        add_tracked_point(tracked_objects, x, y, bounds,
                          sigma_proc, sigma_meas, max_n_coasts=max_n_coasts)

    # Clear the array of observations for the next time step.
    observations[:] = []
    return finished_tracks


class Bounds2D(object):
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def contains(self, x, y):
        return (self.xmin <= x < self.xmax and self.ymin <= y < self.ymax)


class TrackedPoint(object):
    def __init__(self, x, y, vx, vy, sigma_Q, sigma_R):
        self.id = 0
        self.x = x                      # Observed position
        self.y = y
        self.vx = vx                    # Observed instantaneous velocity
        self.vy = vy
        self.boundary = Bounds2D(0, 0, 1000, 1000)
        self.lifetime = 0
        self.n_tail_points = 50
        self.n_coasts = 0
        self.max_n_coasts = 20
        self.coast_length = 0.           # Coast distance so far
        self.max_coast_length = 1000.    # Allowable coast distance
        self.kf = point_2d_kalman_filter([x, vx, y, vy], sigma_Q, sigma_R)
        self.obs_tail = deque()
        self.kf_tail = deque()

    def step_to(self, point):
        """Advance tracker to the observed point.
        point: (x,y) or np.array((x,y))
        """
        self.lifetime += 1

        x, y = point
        self.x, self.y = point

        self.obs_tail.append((x, y))

        if len(self.obs_tail) > self.n_tail_points:
            self.obs_tail.popleft()

        if not self.boundary.contains(x, y):
            return
        else:
            z = np.array([[x], [y]], dtype=np.float32)
            self.kf.predict()
            self.kf.update(z)
        self.update_tail()

    def stay(self, point):
        here = (self.x, self.y)
        self.step_to(here)

    def coast(self):
        self.kf.predict()
        self.update_tail()
        next = np.array((self.kx() + self.kvx(), self.ky() + self.kvy()))
        self.coast_length += norm(next - np.array((self.x, self.y)))
        self.n_coasts += 1
        self.lifetime += 1

    def nearest_observation(self, candidates):
        nearest = np.array((-1, -1))
        here = np.array((self.kx(), self.ky()))
        min_dist = np.inf

        # TODO vectorize
        for i, point in enumerate(candidates):
            dist = norm(np.array(point) - here)
            if dist < min_dist:
                min_dist = dist
                nearest = point
        return nearest, min_dist

    def update_tail(self):
        self.kf_tail.append((self.kx(), self.ky()))
        if len(self.kf_tail) > self.n_tail_points:
            self.kf_tail.popleft()

    def kx(self):
        return self.kf.x[0, 0]

    def kvx(self):
        return self.kf.x[1, 0]

    def ky(self):
        return self.kf.x[2, 0]

    def kvy(self):
        return self.kf.x[3, 0]

    def in_bounds(self):
        return self.boundary.contains(self.x, self.y)

    def coasted_too_long(self):
        return self.n_coasts > self.max_n_coasts

    def coasted_too_far(self):
        return self.coast_length > self.max_coast_length

    def is_valid(self):
        if not self.in_bounds():
            return False
        if self.coasted_too_long():
            return False
        if self.coasted_too_far():
            return False
        return True

    def covariance_ellipse(self):
        """Return center position, 1-sigma axes, and orientation of uncertainty
        ellipse from the x and y components of the state covariance matrix

        Parameters
        ----------
        None

        Returns
        -------
        x, y : float
            center of ellipse
        a, b : float
            semimajor and semiminor axis lengths (1 sigma)
        phi : float
            angle of semimajor axis w.r.t. the x axis in radians
        """
        P = self.kf.P[0:4:2, 0:4:2]  # x-y covariance matrix
        lambdas, vs = eig(P)
        a, b = np.sqrt(lambdas)  # semimajor and semiminor axes
        phi = np.arctan2(vs[1], vs[0])[0]  # ellipse rotation angle
        x, y = self.kx(), self.ky()  # center point
        return x, y, a, b, phi
