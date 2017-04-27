import cv2
import numpy as np
from collections import deque


class Bounds2D:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def contains(self, x, y):
        return (self.xmin <= x < self.xmax and self.ymin <= y < self.ymax)


class TrackedPoint:
    def __init__(self, x0in, y0in, vx0, vy0):
        self.x0 = x0in                  # Position at initialization
        self.y0 = y0in
        self.x = x0in                   # Measured position
        self.y = y0in
        self.kx = x0in                  # Kalman position
        self.ky = y0in
        self.vx = vx0                   # Step velocity dx/dt (= step size).
        self.vy = vy0
        self.kvx = vx0                  # Kalman velocity
        self.kvy = vy0
        self.v = 0.0                    # Historical speed
        self.boundary = Bounds2D(0, 0, 1000, 1000)
        self.lifetime = 0
        self.n_tail_points = 50
        # Number of frames this point "lost the lock" and coasted.
        self.n_coasts = 0
        self.max_n_coasts = 20
        self.coast_length = 0.           # Coast distance so far
        self.max_coast_length = 1000.     # Allowable coast distance
        # 2 state pars, 2 measurement inputs (both x,y)
        self.kf = cv2.KalmanFilter(2, 2)

        proc_var = 1e-4
        meas_var = 1e-3
        err_var = 0.1

        self.kf.transitionMatrix = np.eye(2, dtype=np.float32)
        self.kf.measurementMatrix = np.eye(2, dtype=np.float32)
        self.kf.processNoiseCov = proc_var*np.eye(2, dtype=np.float32)
        self.kf.measurementNoiseCov = meas_var*np.eye(2, dtype=np.float32)
        self.kf.errorCovPost = err_var*np.eye(2, dtype=np.float32)

        self.kf.statePre[0] = self.x
        self.kf.statePre[1] = self.y

        self.obs_tail = deque()
        self.kf_tail = deque()

    def step_to(self, point):
        """
        point: (x,y) or np.array((x,y))
        """
        self.lifetime += 1

        x, y = point
        self.x, self.y = point

        self.obs_tail.append((int(x), int(y)))

        if len(self.obs_tail) > self.n_tail_points:
            self.obs_tail.popleft()

        # Compute historical velocity of this point as the track
        # length divided by # steps
        # if len(self.obs_tail) > 0:
        # tail = np.array(self.obs_tail, dtype=np.int64)
        tail = np.array(self.obs_tail, int)

        self.v = cv2.arcLength(tail, False) / len(tail)

        if not self.boundary.contains(x, y):
            return
        else:
            measurement = np.array([[x], [y]], dtype=np.float32)
            self.kf.predict()
            self.kf.correct(measurement)

            self.kx, self.ky = self.kf.statePost

            self.kf_tail.append((self.kx, self.ky))

            N = len(self.kf_tail)
            if N > self.n_tail_points:
                self.kf_tail.popleft()
            if N > 2:
                tail_x, tail_y = self.kf_tail[N-3]
                self.kvx = (self.kx - tail_x)/2
                self.kvy = (self.ky - tail_y)/2

    def stay(self, point):
        here = (self.x, self.y)
        self.step_to(here)

    def coast(self):
        next = np.array((self.x + self.kvx, self.y + self.kvy))

        self.coast_length += cv2.norm(next - np.array((self.x, self.y)))
        self.step_to(next)
        self.n_coasts += 1

    def nearest_point(self, candidate_points, remove_point=True):
        nearest = np.array((-1, -1))
        here = np.array((self.x, self.y))
        min_dist = np.inf
        i_nearest = -1

        # TODO vectorize
        for i, point in enumerate(candidate_points):
            dist = cv2.norm(np.array(point) - here)
            if dist < min_dist:
                min_dist = dist
                i_nearest, nearest = i, point

        if remove_point and i_nearest >= 0:
            candidate_points.remove(nearest)

        return nearest, min_dist

    def in_bounds(self):
        return self.boundary.contains(self.x, self.y)

    def coasted_too_long(self):
        return self.n_coasts > self.max_n_coasts

    def coasted_too_far(self):
        return self.coast_length > self.max_coast_length
