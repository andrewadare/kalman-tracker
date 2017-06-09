import cv2
import numpy as np
from tracking import TrackedPoint


def add_point(tracked_points, x, y, h, w):
    tp = TrackedPoint(x, y, 0, 0)
    tp.x0, tp.y0 = x, y
    tp.boundary.xmin, tp.boundary.ymin = 0, 0
    tp.boundary.xmax, tp.boundary.ymax = w-1, h-1
    tp.max_coast_length = 0.1*np.sqrt(w*w + h*h)
    tracked_points.append(tp)


def add_sim_point(tracked_points, h, w):
    """
    Note that y increases downward
    """
    v = 5
    dx = np.random.choice([-1, 1])*np.random.uniform(w/50, w/10)
    dy = np.random.uniform(-h/20, h/4)

    x0 = w/2 + dx
    y0 = h/2 + dy
    theta = np.arctan2(dy, dx)
    vx0 = v*np.cos(theta)
    vy0 = v*np.sin(theta)

    tp = TrackedPoint(x0, y0, vx0, vy0)
    tp.boundary.xmin, tp.boundary.ymin = 0, 0
    tp.boundary.xmax, tp.boundary.ymax = w-1, h-1
    tracked_points.append(tp)


def add_noise(x, y, xsigma, ysigma):
    return (x + xsigma*np.random.randn(), y + ysigma*np.random.randn())


def draw(img, counter, sim_points, obs_points):
    img *= 0

    # Draw simulated points
    for i, p in enumerate(sim_points):
        for j, vertex in enumerate(p.obs_tail):
            if j > len(p.obs_tail) - 2:
                break
            next_vertex = p.obs_tail[j+1]
            cv2.line(img, vertex, next_vertex, (100, 100, 100))
        cv2.circle(img, p.obs_tail[-1], 4, (100, 100, 100), -1, 8)

    # Draw observed points
    for i, p in enumerate(obs_points):
        if p.lifetime < 2 or counter < 10:
            continue
        for j, vertex in enumerate(p.obs_tail):
            if j > len(p.obs_tail) - 2:
                break
            next_vertex = p.obs_tail[j+1]
            if counter > 11:
                cv2.line(img, vertex, next_vertex, (0, 0, 255), 2)
        cv2.circle(img, p.obs_tail[-1], 4, (0, 0, 255), -1, 8)

        if counter > 11:
            for j, vertex in enumerate(p.kf_tail):
                if j > len(p.obs_tail) - 2:
                    break
                next_vertex = p.kf_tail[j+1]
                cv2.line(img, vertex, next_vertex, (255, 255, 255), 2)
            cv2.circle(img, p.kf_tail[-1], 4, (255, 255, 255), -1, 8)

    return img


def keep_point(p, frame):
    """
    p: TrackedPoint instance
    frame: image (numpy array)
    """
    if not p.in_bounds():
        return False
    if p.coasted_too_long():
        return False
    if p.coasted_too_far():
        return False
    return True


def main():
    n_points = 10
    h, w = 600, 800
    frame = np.zeros((h, w, 3), np.uint8)

    sim_points = []
    obs_points = []
    measured_positions = []
    xsigma, ysigma = 4., 4.

    cv2.namedWindow('Multi-point tracking simulation')

    frame_index = 0

    while True:
        frame_index += 1

        while len(sim_points) < n_points:
            add_sim_point(sim_points, h, w)

        # Advance simulated points. Simulate position measurements.
        for p in sim_points:
            x = p.x0 + p.vx*p.lifetime
            y = p.y0 + p.vy*p.lifetime
            p.step_to((x, y))

            # Make noisy position measurements
            if p.in_bounds():
                meas_x, meas_y = add_noise(x, y, xsigma, ysigma)

                if p.boundary.contains(meas_x, meas_y):
                    measured_positions.append((meas_x, meas_y))

        sim_points = [p for p in sim_points if p.in_bounds()]

        # Step observations to the nearest available measurement.
        # If no nearby measurement is found in this frame, coast.
        for i, p in enumerate(obs_points):
            nearest, dist = p.nearest_point(measured_positions)
            if dist < 50:
                p.step_to(nearest)
            else:
                p.coast()

        obs_points = [p for p in obs_points if keep_point(p, frame)]

        # Create new tracked observations from any remaining measurements
        for position in measured_positions:
            x, y = position
            add_point(obs_points, x, y, h, w)
        measured_positions = []

        draw(frame, frame_index, sim_points, obs_points)

        cv2.imshow('Simulation', frame)

        if frame_index < 200:
            cv2.imwrite('imgs/{:03d}.png'.format(frame_index), frame)

        k = cv2.waitKey(30) & 0xFF
        if k in [27, 113]:  # esc, q
            break

if __name__ == '__main__':
    main()
