import sys
import cv2
import numpy as np
from tracked_point import match_tracks_to_observations
from tracked_rotated_box import TrackedRotatedBox


def draw_rotated_rect(im, rect, color):
    """
    cv2 is missing a rotated rect drawing function, so work around it.

    Parameters
    ----------
    im : numpy array
        cv2 image
    rect : nested tuple of floats
        ((x,y), (w,h), angle) with angle in degrees
    """
    rbox = cv2.boxPoints(rect)
    cv2.drawContours(im, [np.int0(rbox)], 0, color)


def draw_tracks(im, tracked_boxes):
    red = (0, 0, 255)
    yellow = (0, 240, 240)
    blue = (255, 150, 0)

    # Draw the past few points as a polyline "tail", with the most
    # recent as a circle.
    for i, p in enumerate(tracked_boxes):
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

        # Observed and tracked boxes, respectively
        # cv2.rectangle(im, p.top_left(), p.bottom_right(), blue, 1)
        # cv2.rectangle(im, p.k_top_left(), p.k_bottom_right(), red, 1)
        obs_rect = ((p.x, p.y), (p.w, p.h), p.phi*180/np.pi)
        trk_rect = ((p.kx(), p.ky()), (p.kw(), p.kh()), p.kphi()*180/np.pi)
        draw_rotated_rect(im, obs_rect, blue)
        draw_rotated_rect(im, trk_rect, red)

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


def add_sim_point(tracked_points, h, w, box_size=(100, 100)):
    """
    Add a TrackedPoint instance to the list of simulated ground-truth positions.
    TrackedPoint instances are used for convenience in keeping state. Tracking
    is not done directly on the ground-truth points, but on noisy measurements.
    """
    x, y = np.random.uniform(0, w), np.random.uniform(0, h/4)
    vx, vy = 0, 5
    box_w, box_h = box_size
    phi = np.random.uniform(2*np.pi)
    tp = TrackedRotatedBox([x, vx, y, vy, box_w, box_h, phi], 1., 1.)
    tp.boundary.xmin, tp.boundary.ymin = 0, 0
    tp.boundary.xmax, tp.boundary.ymax = w-1, h-1
    tracked_points.append(tp)


def step(sim_points):
    """Advance simulated points and remove any that step out of bounds.
    The `sim_points` array is modified in place.
    Parameters
    ----------
    sim_points : sequence of TrackedPoint objects
    """

    # Made-up perturbations to slowly change box shape/size and rotation.
    fw = 0.8*np.cos(0.05*step.i)
    fh = 0.4*np.sin(0.01*step.i)
    fp = 0.05*np.sin(0.01*step.i)

    for p in sim_points:
        p.step((p.x + p.vx, p.y + p.vy, p.w + fw, p.h + fh, p.phi + fp))

    # Remove out-of-bounds points
    sim_points[:] = [p for p in sim_points if p.in_bounds()]
    step.i += 1
    return sim_points


step.i = 0


def add_noise(x, y, xsigma, ysigma):
    return (x + xsigma*np.random.randn(), y + ysigma*np.random.randn())


def observe(sim_points, observations, xsigma, ysigma, miss_prob=0.1):
    for p in sim_points:
        if np.random.uniform() > 1.0 - miss_prob:
            continue
        meas_x, meas_y = add_noise(p.x, p.y, xsigma, ysigma)
        box_w = p.w + 0.1*np.random.randn()
        box_h = p.h + 0.1*np.random.randn()
        phi = p.phi + 0.05*np.random.randn()
        if p.boundary.contains(meas_x, meas_y):
            observations.append((meas_x, meas_y, box_w, box_h, phi))


def main(save_imgs=False):
    n_points = 5
    h, w = 800, 600
    bounds = (0, 0, w, h)
    im = np.zeros((h, w, 3), np.uint8)

    sim_points = []  # simulated true positions
    observations = []  # noisy measurements: box center x,y and w,h
    tracked_objects = []  # tracks
    xsigma, ysigma = 0.002*w, 0.002*h
    sigma_proc = 1  # Smaller sigma_Q --> smoother but more sluggish
    sigma_meas = 5
    max_n_coasts = 3

    cv2.namedWindow('Simulation')

    def add_track(tracked_objects, observation):
        """Append a new TrackedPoint object to the `tracked_objects` list.
        This is used as a callback in match_tracks_to_observations and the
        signature should not change.

        Parameters
        ----------
        tracked_objects : list(TrackedBox)
            Append a new TrackedPoint to this list
        observation : sequence of floats
            (center x, center y, box width, box height)
        """
        x, y, w, h, phi = observation
        tp = TrackedRotatedBox([x, 0, y, 0, w, h, phi], sigma_proc, sigma_meas)

        xmin, ymin, xmax, ymax = bounds
        tp.boundary.xmin, tp.boundary.ymin = xmin, ymin
        tp.boundary.xmax, tp.boundary.ymax = xmax, ymax
        tp.max_n_coasts = max_n_coasts
        tp.id = add_track.id
        add_track.id += 1
        tracked_objects.append(tp)

    add_track.id = 0

    frame_index = 0
    while True:
        step(sim_points)

        while len(sim_points) < n_points:
            add_sim_point(sim_points, h, w)

        observe(sim_points, observations, xsigma, ysigma)

        match_tracks_to_observations(tracked_objects, observations, add_track)

        # Filter out lost/out-of-bounds tracks
        tracked_objects = [t for t in tracked_objects if t.is_valid()]

        # Set delay to 0 to step on keypress. Press ESC or q to quit.
        visualize(im, sim_points, tracked_objects, delay=0)

        if save_imgs and frame_index < 250:
            name = 'imgs/rbb{:03d}.png'.format(frame_index)
            cv2.imwrite(name, im)
            print('saved', name)
        frame_index += 1


if __name__ == '__main__':
    # To save an animated GIF using the ImageMagick convert tool:
    # convert -delay 30 -loop 0 imgs/rbb*.png tracked_rotated_boxes.gif
    main(save_imgs=False)

