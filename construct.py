import numpy as np


class Environment:
    """

    """
    camera_to_wall_distance = 3  # m
    camera_resolution = (800, 600)  # (width, height) px


class Servo:
    """

    """
    __angle_bounds = (0, 180)  # degrees
    __default_angle = 90  # degrees
    __speed = 10  # deg/tick
    __slippage = 0  # deg (UNIMPLEMENTED)

    def __init__(self):
        self._angle = self.__default_angle

    # noinspection PyMethodParameters
    def __enforce_bounds(foo):
        # this is obviously a decorator (go home PyCharm, you drunk)
        def wrap(self, angle):
            angle = max(min(angle, self.__angle_bounds[1]), self.__angle_bounds[0])
            out = foo(self, angle)
            return out

        return wrap

    def get_settings(self):
        return self.__angle_bounds, self.__default_angle, self.__speed, self.__slippage

    def get_angle(self):
        return self._angle

    @__enforce_bounds
    def set_angle(self, angle):
        self._angle = angle

    @__enforce_bounds
    def move_to(self, angle):
        if angle == self._angle:
            pass
        elif angle < self._angle:
            self._angle = max(self._angle - self.__speed, angle)
        else:
            self._angle = min(self._angle + self.__speed, angle)

    @__enforce_bounds
    def get_dot_wall_position(self, angle):
        """ calculate position of laser dot on wall based on goniometric functions
        and angles of the servo

        :param angle: (int) angle of servo which is perpendicular to wall [degrees]
        :return position: (float) position of dot [meters]
        """
        wall_dist = Environment.camera_to_wall_distance

        return np.tan(np.deg2rad(self.__default_angle - angle))*wall_dist
        # TODO: convert to pixel equivalent with given camera resolution


class Laser:
    """

    """
    def __init__(self, servos=(Servo(), Servo())):
        self._servo_x = servos[0]
        self._servo_y = servos[1]

        self.angle_x = servos[0].get_angle()
        self.angle_y = servos[1].get_angle()

        self.wall_pos_x = servos[0].get_dot_wall_position(self.angle_x)
        self.wall_pos_y = servos[1].get_dot_wall_position(self.angle_y)

    def move_x_y_tick(self, angle_x: int = None, angle_y: int = None):
        """ move servos to specific angle, with consideration of servo speed and update the current wall positions

        :param angle_x: (int) angle of servo moving along x (horizontal) axis (if None, keep last position)
        :param angle_y: (int) angle of servo moving along y (vertical) axis
        """
        if angle_x is None:
            angle_x = self.angle_x
        if angle_y is None:
            angle_y = self.angle_y

        self._servo_x.move_to(angle_x)
        self._servo_y.move_to(angle_y)

        self.angle_x = self._servo_x.get_angle()
        self.angle_y = self._servo_y.get_angle()

        # update wall positions
        self.wall_pos_x = self._servo_x.get_dot_wall_position(self.angle_x)
        self.wall_pos_y = self._servo_y.get_dot_wall_position(self.angle_y)

class Construct:
    """

    """
    __vertical_laser_distance = 0.2  # m

    def __init__(self, lasers=(Laser(), Laser())):
        self._laser_red = lasers[0]
        self._laser_green = lasers[1]

    def run(self):
        pass
