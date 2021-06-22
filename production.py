import time
import numpy as np
import itertools

from construct import Environment, Servo, Laser, Construct, Wall, PathGenerator

from matplotlib import pyplot as plt


if __name__ == '__main__':
    laser_red = Laser()
    laser_green = Laser()
    wall = Wall()
    generator = np.random.default_rng()
    path_gen = PathGenerator()
    env = Environment()

    row_angles = list(range(50, 131, 10))
    column_angles = np.array([[d]*len(row_angles) for d in row_angles]).flatten()

    print(column_angles)

    angles = itertools.cycle(column_angles)
    angles2 = itertools.cycle(row_angles)

    xs, ys = path_gen.ellipse(scale=0.8, circle=True)

    # convert pixel path to servo angles
    aov, fov = env.get_fov()
    ppm_x = env.get_ppm(fov[0], axis=0)
    ppm_y = env.get_ppm(fov[1], axis=1)

    angles_ellipse = []
    ya = 90
    for x, y in zip(xs, ys):
        xm = env.pixel_to_meter(x, ppm_x, axis=0)
        ym = env.pixel_to_meter(y, ppm_y, axis=1)
        xa = env.meter_to_angle(xm, ya)
        ya = env.meter_to_angle(ym, xa)
        angles_ellipse.append((xa, ya))
        # TODO: check if angles are calculated correctly

    angles_ellipse = itertools.cycle(angles_ellipse)

    for (ax, ay), a1, a2 in zip(angles_ellipse, angles, angles2):
        # ang = generator.integers(50, 130, size=4)

        done_red = False
        done_green = False
        j = 0

        while not done_red and not done_green:
            done_red = laser_red.move_x_y_tick(ax, a)
            done_green = laser_green.move_x_y_tick(a2, a1)
            red_pos = (laser_red.wall_pos_x, laser_red.wall_pos_y)
            green_pos = (laser_green.wall_pos_x, laser_green.wall_pos_y)
            j += 1
            wall.update(red_pos, green_pos)
            time.sleep(0.05)

        # print([abs(r - g) for r, g in zip(red_pos, green_pos)])
        # DONE: test, if the cos(Beta) in calculating wall position is actually correct!!!!
        # DONE: the lasers seem to move together??? -- SOLVED
        # TODO: move this to Construct class when it works