import sys
import cv2
import numpy as np
from tracking import TrackedPoint


def add_tracked_point(tracked_points, x, y, bounds,
                      sigma_proc, sigma_meas, max_n_coasts=3):
    """
    Parameters
    ----------
    tracked_points : list(TrackedPoint)
        Append a new TrackedPoint to this list
    x, y : float
        Initial position
    bounds : sequence of floats
        (xmin, ymin, xmax, ymax)
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


def draw_tracks(im, tracked_points):
    red = (0, 0, 255)
    yellow = (0, 240, 240)
    blue = (255, 150, 0)

    # Draw the past few points as a polyline "tail", with the most
    # recent as a circle.
    for i, p in enumerate(tracked_points):
        if p.lifetime == 0:
            continue
        # Observed positions
        for j, vertex in enumerate(p.obs_tail):
            if j > len(p.obs_tail) - 2:
                break
            vertex = tuple([int(x) for x in vertex])
            next_vertex = tuple([int(x) for x in p.obs_tail[j+1]])
            cv2.line(im, vertex, next_vertex, blue, 2)
        if len(p.obs_tail) > 0:
            x, y = p.obs_tail[-1]
            cv2.circle(im, (int(x), int(y)), 4, blue, -1, 4)

        # Tracked positions
        for j, vertex in enumerate(p.kf_tail):
            if j > len(p.kf_tail) - 2:
                break
            vertex = tuple([int(x) for x in vertex])
            next_vertex = tuple([int(x) for x in p.kf_tail[j+1]])
            cv2.line(im, vertex, next_vertex, red, 2)
        cv2.circle(im, (int(p.kx()), int(p.ky())), 3, red, -1, 3)

        x, y, a, b, phi = p.covariance_ellipse()
        center = (int(x), int(y))
        axes = (int(3*a), int(3*b))  # 3 sigma contours
        angle = int(phi*180/np.pi)
        cv2.ellipse(im, center, axes, angle, 0, 360, yellow, 2)
    return im


def visualize(im, sim_points, tracked_objects, delay=30):
    im *= 0  # clear the canvas
    for p in sim_points:
        pos = (int(p.x), int(p.y))
        cv2.circle(im, pos, 4, (255, 255, 255), -1, 4)
    draw_tracks(im, tracked_objects)
    cv2.imshow('Simulation', im)
    k = cv2.waitKey(delay)
    if k in [27, 113]:  # esc, q
        sys.exit(0)


def add_sim_point(tracked_points, h, w):
    """
    Add a TrackedPoint instance to the list of simulated ground-truth positions.
    TrackedPoint instances are used for convenience in keeping state. Tracking
    is not done directly on the ground-truth points, but on noisy measurements.
    """
    x, y = np.random.uniform(0, w), np.random.uniform(0, h/4)
    vx, vy = 0, 5
    tp = TrackedPoint(x, y, vx, vy, 1., 1.)
    tp.boundary.xmin, tp.boundary.ymin = 0, 0
    tp.boundary.xmax, tp.boundary.ymax = w-1, h-1
    tracked_points.append(tp)


def match_tracks_to_observations(tracked_objects, observations, bounds,
                                 distance_threshold=30):
    """Associate tracks to observations. The `tracked_objects` and
    `observations` lists are modified in-place.

    Parameters
    ----------
    tracked_objects : sequence of TrackedPoint instances
    observations : sequence of int or float pairs
        observed (x,y) positions
    bounds : sequence of floats
        (xmin, ymin, xmax, ymax)
    distance_threshold : int
        Maximum distance to nearest observation for matching

    Returns
    -------
    None
    """

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

    # Filter out lost/out-of-bounds tracks
    tracked_objects[:] = [d for d in tracked_objects if d.is_valid()]

    # Start tracking any remaining measurements under the assumption
    # that they are new (not yet tracked).
    for i, observation in enumerate(observations):
        x, y = observation
        add_tracked_point(tracked_objects, x, y, bounds, 0.1, 10, max_n_coasts=3)
        # add_tracked_point(tracked_objects, x, y, bounds, 0.001, 0.1, max_n_coasts=3)

    # Clear the array of observations for the next time step.
    observations[:] = []


def step(sim_points):
    """Advance simulated points and remove any that step out of bounds.
    The `sim_points` array is modified in place.
    Parameters
    ----------
    sim_points : sequence of TrackedPoint objects
    """
    for p in sim_points:
        p.step_to((p.x + p.vx, p.y + p.vy))

    # Remove out-of-bounds points
    sim_points[:] = [p for p in sim_points if p.in_bounds()]

    return sim_points


def add_noise(x, y, xsigma, ysigma):
    return (x + xsigma*np.random.randn(), y + ysigma*np.random.randn())


def observe(sim_points, observations, xsigma, ysigma, miss_prob=0.1):
    for p in sim_points:
        if np.random.uniform() > 1.0 - miss_prob:
            continue
        meas_x, meas_y = add_noise(p.x, p.y, xsigma, ysigma)
        if p.boundary.contains(meas_x, meas_y):
            observations.append((meas_x, meas_y))


def main():
    n_points = 10
    h, w = 800, 600
    bounds = (0, 0, w, h)
    im = np.zeros((h, w, 3), np.uint8)

    sim_points = []  # simulated true positions
    observations = []  # noisy measurements
    tracked_objects = []  # tracks
    xsigma, ysigma = 0.005*w, 0.005*h

    cv2.namedWindow('Simulation')

    loop_index = 0
    while True:
        step(sim_points)

        while len(sim_points) < n_points:
            add_sim_point(sim_points, h, w)

        observe(sim_points, observations, xsigma, ysigma)

        match_tracks_to_observations(tracked_objects, observations, bounds)

        # Filter out lost/out-of-bounds tracks
        tracked_objects = [t for t in tracked_objects if t.is_valid()]

        # Set delay to 0 to step on keypress. Press ESC or q to quit.
        visualize(im, sim_points, tracked_objects, delay=0)
        loop_index += 1

if __name__ == '__main__':
    main()
