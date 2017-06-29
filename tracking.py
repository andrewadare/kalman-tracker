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
    def __init__(self, x, y, vx, vy):
        self.x = x                   # Observed position
        self.y = y
        self.kx = x                  # Kalman position
        self.ky = y
        self.vx = vx                   # Step velocity dx/dt (= step size).
        self.vy = vy
        self.kvx = vx                  # Kalman velocity
        self.kvy = vy
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
        err_var = 1e-1

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
        self.kf.predict()
        # self.kf.correct(measurement)

        self.kx, self.ky = self.kf.statePre
        self.x, self.y = self.kf.statePre

        self.kf_tail.append((self.kx, self.ky))

        N = len(self.kf_tail)
        if N > self.n_tail_points:
            self.kf_tail.popleft()
        if N > 2:
            tail_x, tail_y = self.kf_tail[N-3]
            self.kvx = (self.kx - tail_x)/2
            self.kvy = (self.ky - tail_y)/2

        next = np.array((self.x + self.kvx, self.y + self.kvy))

        self.coast_length += cv2.norm(next - np.array((self.x, self.y)))
        # self.step_to(next)
        self.n_coasts += 1
        self.lifetime += 1

    def nearest_observation(self, candidates):
        nearest = np.array((-1, -1))
        here = np.array((self.x, self.y))
        min_dist = np.inf

        # TODO vectorize
        for i, point in enumerate(candidates):
            dist = cv2.norm(np.array(point) - here)
            if dist < min_dist:
                min_dist = dist
                nearest = point
        return nearest, min_dist

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
