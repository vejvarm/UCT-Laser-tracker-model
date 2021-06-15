import time
import numpy as np

from construct import Environment, Servo, Laser, Construct, Wall

from matplotlib import pyplot as plt


if __name__ == '__main__':
    laser_red = Laser()
    laser_green = Laser()
    wall = Wall()
    generator = np.random.default_rng()

    for i in range(1000):
        rng_angles = generator.integers(50, 130, size=4)

        done_red = False
        done_green = False
        j = 0

        print(rng_angles)

        while j < 5 and (not done_red and not done_green):
            done_red = laser_red.move_x_y_tick(rng_angles[0], rng_angles[1])
            done_green = laser_green.move_x_y_tick(rng_angles[2], rng_angles[3])
            red_pos = (laser_red.wall_pos_x, laser_red.wall_pos_y)
            green_pos = (laser_green.wall_pos_x, laser_green.wall_pos_y)
            wall.update(red_pos, green_pos)
            j += 1
            time.sleep(0.01)

        print(red_pos, green_pos)
        # TODO: test, if the cos(Beta) in calculating wall position is actually correct!!!!
        # TODO: the lasers seem to move together???
        # TODO: move this to Construct class when it works