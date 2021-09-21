import time
import numpy as np
import itertools

from environment import Transformations, Servo, Laser, Construct, Wall, PathGenerator

from matplotlib import pyplot as plt


def generate_ellipse_angles(xs, ys, env):
    # convert pixel path to servo angles
    aov, fov = env.get_fov()
    ppm_x = env.get_ppm(fov[1], axis=1)
    ppm_y = env.get_ppm(fov[0], axis=0)

    xas, yas = list(), list()
    ya = 90
    for x, y in zip(xs, ys):
        xm = env.pixel_to_meter(x, ppm_x, axis=0)
        ym = env.pixel_to_meter(y, ppm_y, axis=1)
        xa = env.meter_to_angle(xm, ya)
        ya = env.meter_to_angle(ym, xa)
        xas.append(xa)
        yas.append(ya)

    return xas, yas


if __name__ == '__main__':
    laser_red = Laser()
    laser_green = Laser()
    wall = Wall(blit=True)
    generator = np.random.default_rng()
    path_gen = PathGenerator()
    env = Transformations()

    row_angles = list(range(50, 131, 10))
    column_angles = np.array([[d]*len(row_angles) for d in row_angles]).flatten()

    angles = itertools.cycle(column_angles)
    angles2 = itertools.cycle(row_angles)

    xas_red, yas_red = path_gen.ellipse(scale=0.9, resolution=0.05*np.pi, circle=True, return_angles=True)
    xas_green, yas_green = path_gen.ellipse(scale=0.5, resolution=0.05*np.pi, circle=True, return_angles=True)

    for xa_red, ya_red, xa_green, ya_green in zip(xas_red, yas_red, xas_green, yas_green):
        # ang = generator.integers(50, 130, size=4)

        done_red = False
        done_green = False
        j = 0

        while not done_red and not done_green:
            done_red = laser_red.move_x_y_tick(xa_red, ya_red)
            done_green = laser_green.move_x_y_tick(xa_green, ya_green)
            red_pos = (laser_red.wall_pos_x, laser_red.wall_pos_y)
            green_pos = (laser_green.wall_pos_x, laser_green.wall_pos_y)
            wall.update(red_pos, green_pos)
            time.sleep(0.05)

        # print([abs(r - g) for r, g in zip(red_pos, green_pos)])
        # DONE: test, if the cos(Beta) in calculating wall position is actually correct!!!!
        # DONE: the lasers seem to move together??? -- SOLVED
        # DONE: move this to Construct class when it works