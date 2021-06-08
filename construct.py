import numpy as np


class Environment:
    """

    """
    camera_resolution = (4056, 3040)  # (width, height) px
    camera_to_wall_distance = 4.  # m
    _chip_size = (7.564, 5.476)  # (width, height) mm
    _crop_factor = 5.54
    _lens_focal_length = (2.8, 12)  # mm

    def __init__(self):
        pass

    def get_fov(self, c2w_distance=camera_to_wall_distance, focal_length=_lens_focal_length[0]):
        """ calculate camera fov based on camera to wall distance and lens adjusted focal length

        :param c2w_distance: (float) distance from camera to projection wall in [m]
        :param focal_length: (float) lens focal length in [mm] (not adjusted to crop factor)
        :return aov, fov: Tuple[float], Tuple[float] Width x Height angle of view [deg] and field of view of the camera [m]
        """

        aov = [2*np.arctan(s/(2*focal_length)) for s in self._chip_size]  # angle of view (rad)
        fov = [2*c2w_distance*np.tan(a/2) for a in aov]  # field of view

        return tuple(np.rad2deg(aov)), tuple(fov)

    def get_ppm(self, fov: float, axis=0):
        """

        :param fov: (float) field of view in meters
        :param axis: (0) axis for which fov is calculated | Horizontal (0) | Vertical (1) |
        :return ppm: (float) pixels per meter
        """

        resolution = self.camera_resolution[int(axis)]

        ppm = resolution/fov

        return ppm


class Servo:
    """

    """
    __angle_bounds = (0, 180)  # degrees
    __default_angle = 90  # degrees
    __speed = 10  # deg/tick
    __slippage = 0  # deg (UNIMPLEMENTED)

    def __init__(self):
        self._angle = self.__default_angle
        self.e = Environment()

    # noinspection PyMethodParameters
    def __enforce_bounds(foo):
        # this is obviously a decorator (go home PyCharm, you drunk)
        def wrap(self, angle, *args, **kwargs):
            angle = max(min(angle, self.__angle_bounds[1]), self.__angle_bounds[0])
            out = foo(self, angle, *args, **kwargs)
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
    def get_dot_wall_position(self, angle, fov, axis=0):
        """ calculate position of laser dot on wall based on goniometric functions
        and angles of the servo

        :param angle: (int) angle of servo which is perpendicular to wall [degrees]
        :param fov: (float) field of view of the camera for the given axis [meters]
        :param axis: (int) either horizontal (0) or vertical (1) axis
        :return position: Tuple[int] position of dot [pixels]
        """
        wall_dist = self.e.camera_to_wall_distance
        pixel_resolution = self.e.camera_resolution

        # aov, fov = self.e.get_fov()

        meter_distance = np.tan(np.deg2rad(self.__default_angle - angle))*wall_dist

        ppm = self.e.get_ppm(fov, axis)  # pixels per meter

        pixel_distance = pixel_resolution[axis]/2 + ppm*meter_distance

        return int(pixel_distance)
        # TODO: TEST if correct! convert to pixel equivalent with given camera resolution
        # TODO: Ensure boundaries
        # TODO: Make simple GUI for showing 2D position


class Laser:
    """

    """
    def __init__(self, env=Environment(), servos=(Servo(), Servo())):
        self._env = env

        self._servo_x = servos[0]
        self._servo_y = servos[1]

        self.angle_x = servos[0].get_angle()
        self.angle_y = servos[1].get_angle()

        self.aov, self.fov = self._env.get_fov()  # (x, y) [deg], (x, y) [m]

        self.wall_pos_x = servos[0].get_dot_wall_position(self.angle_x, self.fov[0], axis=0)
        self.wall_pos_y = servos[1].get_dot_wall_position(self.angle_y, self.fov[1], axis=1)

    def set_fov(self, fov: tuple):
        self.fov = fov

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
        self.wall_pos_x = self._servo_x.get_dot_wall_position(self.angle_x, self.fov[0], axis=0)
        self.wall_pos_y = self._servo_y.get_dot_wall_position(self.angle_y, self.fov[0], axis=1)

        if angle_x == self.angle_x and angle_y == self.angle_y:
            return True
        else:
            return False


class Construct:
    """

    """
    __vertical_laser_distance = 0.2  # m

    def __init__(self, lasers=(Laser(), Laser())):
        self._laser_red = lasers[0]
        self._laser_green = lasers[1]

    def run(self):
        pass
