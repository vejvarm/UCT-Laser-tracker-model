from abc import ABC

import numpy as np

from collections import namedtuple

from matplotlib import pyplot as plt

from helpers import generate_path

from transformations import Transformations

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


class Servo:
    """

    """
    __speed = 10  # deg/tick
    __slippage = 0  # deg (UNIMPLEMENTED)

    _angle_bounds = (0, 180)  # degrees
    _default_angle = 90  # degrees

    def __init__(self):
        self._angle = self._default_angle
        self.e = Transformations()

    # noinspection PyMethodParameters
    def __enforce_bounds(foo):
        # this is obviously a decorator (go home PyCharm, you drunk)
        def wrap(self, angle, *args, **kwargs):
            angle = max(min(angle, self._angle_bounds[1]), self._angle_bounds[0])
            out = foo(self, angle, *args, **kwargs)
            return out

        return wrap

    def get_settings(self):
        return self._angle_bounds, self._default_angle, self.__speed, self.__slippage

    def get_angle(self):
        return self._angle

    def get_default_angle(self):
        return self._default_angle

    @__enforce_bounds
    def set_angle(self, angle):
        self._angle = int(angle)

    @__enforce_bounds
    def set_default_angle(self, angle):
        self._default_angle = angle

    @__enforce_bounds
    def move_to(self, angle):
        if angle == self._angle:
            pass
        elif angle < self._angle:
            self._angle = int(max(self._angle - self.__speed, angle))
        else:
            self._angle = int(min(self._angle + self.__speed, angle))

    @__enforce_bounds
    def get_dot_wall_position(self, angle, fov, axis=0, angle2=90, angle2_default=90):
        """ calculate position of laser dot on wall based on goniometric functions
        and angles of the servo

        :param angle: (int) angle of servo which is perpendicular to wall [degrees]
        :param fov: (float) field of view of the camera for the given axis [meters]
        :param axis: (int) either horizontal (0) or vertical (1) axis
        :param angle2: (int) angle of the secondary axis (if 0, it has no effect) [degrees]
        :param angle2_default: (int) default angle of the secondary axis [degrees]
        :return position: Tuple[int] position of dot [pixels]
        """
        pos_meters = self.e.angle_to_meter(angle, angle2, (self._default_angle, angle2_default))

        ppm = self.e.get_ppm(fov, axis)  # pixels per meter

        pos_pixels = self.e.meter_to_pixel(pos_meters, ppm, axis)

        return pos_pixels
        # DONE: TEST if correct! convert to pixel equivalent with given camera resolution
        # TODO: Ensure boundaries


class Laser:
    """

    """
    def __init__(self, env=Transformations(), servos=(None, None)):
        self._env = env

        if servos[0] is None:
            self._servo_x = Servo()
        else:
            self._servo_x = servos[0]
        if servos[1] is None:
            self._servo_y = Servo()
        else:
            self._servo_y = servos[1]

        self.default_angle_x = self._servo_x.get_default_angle()
        self.default_angle_y = self._servo_y.get_default_angle()

        self.angle_x = self._servo_x.get_angle()
        self.angle_y = self._servo_y.get_angle()

        self.aov, self.fov = self._env.get_fov()  # (x, y) [deg], (x, y) [m]

        self.wall_pos_x = self._servo_x.get_dot_wall_position(self.angle_x,
                                                              self.fov[0],
                                                              axis=0,
                                                              angle2=self.angle_y,
                                                              angle2_default=self.default_angle_y)
        self.wall_pos_y = self._servo_y.get_dot_wall_position(self.angle_y,
                                                              self.fov[1],
                                                              axis=1,
                                                              angle2=self.angle_x,
                                                              angle2_default=self.default_angle_x)

    def set_fov(self, fov: tuple):
        self.fov = fov

    def move_angle_tick(self, angle_x: int = None, angle_y: int = None, speed_restrictions=True):
        """ move servos to specific angle, with consideration of servo speed and update the current wall positions

        :param angle_x: (int) angle of servo moving along x (horizontal) axis (if None, keep last position)
        :param angle_y: (int) angle of servo moving along y (vertical) axis
        :param speed_restrictions: (bool) if False, angle increments are not restricted by servo speed
        """
        if angle_x is None:
            angle_x = self.angle_x
        if angle_y is None:
            angle_y = self.angle_y

        if speed_restrictions:
            self._servo_x.move_to(angle_x)
            self._servo_y.move_to(angle_y)
        else:
            self._servo_x.set_angle(angle_x)
            self._servo_y.set_angle(angle_y)

        self.default_angle_x = self._servo_x.get_default_angle()
        self.default_angle_y = self._servo_y.get_default_angle()

        self.angle_x = self._servo_x.get_angle()
        self.angle_y = self._servo_y.get_angle()

        # update wall positions
        self.wall_pos_x = self._servo_x.get_dot_wall_position(self.angle_x,
                                                              self.fov[0],
                                                              axis=0,
                                                              angle2=self.angle_y,
                                                              angle2_default=self.default_angle_y)
        self.wall_pos_y = self._servo_y.get_dot_wall_position(self.angle_y,
                                                              self.fov[1],
                                                              axis=1,
                                                              angle2=self.angle_x,
                                                              angle2_default=self.default_angle_x)

        if angle_x == self.angle_x and angle_y == self.angle_y:
            return True
        else:
            return False


class LaserTracker(py_environment.PyEnvironment, ABC):

    def __init__(self, lasers=(None, None), visualize=False, speed_restrictions=True, steps_per_ep=1000,
                 angle_bounds=(0, 180), max_angle_step=10, target_path="random"):
        super().__init__()
        self.env = Transformations()

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(2, ), dtype=np.float32, minimum=angle_bounds[0], maximum=angle_bounds[1], name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(4,), dtype=np.float32, minimum=0, maximum=1., name='observation')
        self._episode_ended = False
        self.speed_restrictions = speed_restrictions
        self._steps_per_ep = steps_per_ep
        self._step_counter = 0
        self._reward_sum = 0.

        if lasers[0] is None:
            self._laser_red = Laser()
        else:
            self._laser_red = lasers[0]
        if lasers[1] is None:
            self._laser_green = Laser()
        else:
            self._laser_green = lasers[1]

        self.visualize = visualize

        self.green_pos = (self._laser_green.wall_pos_x, self._laser_green.wall_pos_y)
        self.red_pos = (self._laser_red.wall_pos_x, self._laser_red.wall_pos_y)

        self._observation = np.zeros(shape=4, dtype=np.float32)
        self._old_observation = np.zeros_like(self._observation)
        self._observation[:] = self.env.normalize_obs([*self.green_pos, *self.red_pos])  # normalize to [0,1) float
        self._old_observation[:] = self.env.normalize_obs([*self.green_pos, *self.red_pos])  # normalize to [0,1) float

        if self.visualize:
            self.wall = Wall(blit=True)
        else:
            self.wall = None

        self.default_path = generate_path(path_type=target_path,
                                          angle_bounds=angle_bounds,
                                          max_angle_step=max_angle_step)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._observation[:] = self.env.normalize_obs([*self.green_pos, *self.red_pos])  # normalize to [0,1) float
        self._step_counter = 0
        self._reward_sum = 0.
        self._episode_ended = False
        return ts.restart(self._observation)

    def _step(self, action):
        self._step_counter += 1

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # move green laser based on agent inputs
        # print(action)
        _ = self._laser_green.move_angle_tick(action[0], action[1], self.speed_restrictions)
        self.green_pos = (self._laser_green.wall_pos_x, self._laser_green.wall_pos_y)

        # move red laser based on path from path_gen
        x_red, y_red = next(self.default_path)
        done = self._laser_red.move_angle_tick(x_red, y_red, self.speed_restrictions)
        # rebase observation with new red position
        self.red_pos = (self._laser_red.wall_pos_x, self._laser_red.wall_pos_y)

        # update observation
        self._observation[:] = self.env.normalize_obs([*self.green_pos, *self.red_pos])  # normalize to [0,1) float

        # End episode after self._steps_per_ep
        if self._step_counter >= self._steps_per_ep:
            self._episode_ended = True

        # calculate reward
        reward = self.env.reward(np.expand_dims(self._observation, 0), np.expand_dims(self._old_observation, 0))
        self._reward_sum += reward

        # update old observation
        self._old_observation[:] = [*self._observation]  # normalize to [0,1) float

        if self._episode_ended:
            print(self._reward_sum/self._step_counter)
            return ts.termination(self._observation, reward)
        else:
            return ts.transition(self._observation, reward=reward, discount=.99)


class Wall:
    """

    """

    def __init__(self, resolution=Transformations.camera_resolution, blit=False):
        self.resolution = resolution
        self.blit = blit
        self.fig, self.ax = plt.subplots()

        self.ax.set_xlim([0, self.resolution[0]])
        self.ax.set_ylim([0, self.resolution[1]])

        center = [r//2 for r in self.resolution]

        self.stm_red = self.ax.stem([center[0]], [center[1]], linefmt="none", markerfmt="rx", basefmt="none")
        self.stm_grn = self.ax.stem([center[0]], [center[1]], linefmt="none", markerfmt="go", basefmt="none")

        self.fig.canvas.draw()

        if self.blit:
            # cache the background
            self.bcgrd = self.fig.canvas.copy_from_bbox(self.ax.bbox)

        plt.show(block=False)

    def update(self, red_pos=(0, 0), grn_pos=(0, 0)):
        """ one step of wall update to show current position of lasers

        :param red_pos: Tuple[int] x and y position of the red laser
        :param grn_pos: Tuple[ind] x and y position of the green laser
        """
        # set new data
        if 0 < red_pos[0] < self.resolution[0] and 0 < red_pos[1] < self.resolution[1]:
            self.stm_red.markerline.set_data(red_pos[0], red_pos[1])
        if 0 < grn_pos[0] < self.resolution[0] and 0 < grn_pos[1] < self.resolution[1]:
            self.stm_grn.markerline.set_data(grn_pos[0], grn_pos[1])

        if self.blit:
            # restore background
            self.fig.canvas.restore_region(self.bcgrd)

            # draw artists
            self.ax.draw_artist(self.stm_red.markerline)
            self.ax.draw_artist(self.stm_grn.markerline)

            # fill in the axes rectangle
            self.fig.canvas.blit(self.ax.bbox)
        else:
            self.fig.canvas.draw()

        self.fig.canvas.flush_events()

