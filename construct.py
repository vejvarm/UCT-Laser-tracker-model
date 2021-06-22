import numpy as np
from matplotlib import pyplot as plt


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
        :param axis: (int) axis for which fov is calculated | Horizontal (0) | Vertical (1) |
        :return ppm: (float) pixels per meter
        """

        resolution = self.camera_resolution[int(axis)]

        ppm = resolution/fov

        return ppm

    def meter_to_pixel(self, pos_meters: float, ppm: float, axis: int):
        """

        :param pos_meters: (float) position of laser point from center in [meters]
        :param ppm: (float) number of pixels per one meter
        :param axis: (int) axis for which conversion is calculated | Horizontal (0) | Vertical (1) |
        :return pos_pixels: (int) position of laser point from center in [pixels]
        """
        return int(np.round(self.camera_resolution[axis]/2 + ppm * pos_meters))

    def pixel_to_meter(self, pos_pixels: int, ppm: float, axis: int):
        """

        :param pos_pixels: (int) position of laser point from center in [pixels]
        :param ppm: (float) number of pixels per one meter
        :param axis: (int) axis for which conversion is calculated | Horizontal (0) | Vertical (1) |
        :return pos_meters: (float) position of laser point from center in [meters]
        """
        return (pos_pixels-self.camera_resolution[axis]/2)/ppm

    def angle_to_meter(self, angle, angle2=90, angle_defaults=(90, 90)):
        """ convert servo angle to meter distance from _default_angle position ("center")

        :param angle: (int) current angle of the servo
        :param angle2: (int) angle of the secondary axis (if 90, it has no effect) [degrees]
        :param angle_defaults: (Tuple[int]) default angles of both axes [degrees]
        :return meter_pos: (float) distance of laser point from "center" in meters
        """
        wall_dist = self.camera_to_wall_distance
        alpha = np.deg2rad(angle_defaults[0] - angle)
        beta = np.deg2rad(angle_defaults[1] - angle2)
        return wall_dist*np.tan(alpha)/np.cos(beta)

    def meter_to_angle(self, dist_meters, angle2=90, angle_defaults=(90, 90)):
        """ convert meter position on wall to angle distance from _default_angle position ("center")

        :param dist_meters: (float) distance of laser point from "center" in meters
        :param angle2: (int) angle of the secondary axis (if 90, it has no effect) [degrees]
        :param angle_defaults: (Tuple[int]) default angles of both axes [degrees]
        :return angle: (int) respective angle of the servo
        """
        wall_dist = self.camera_to_wall_distance  # [meters]
        beta = np.deg2rad(angle_defaults[1] - angle2)  # [radians]
        return int(np.round(angle_defaults[0] - np.rad2deg(np.arctan(dist_meters*np.cos(beta)/wall_dist))))


class PathGenerator:

    def __init__(self, resolution=Environment.camera_resolution):
        self.resolution = resolution

    def ellipse(self, scale=0.5, resolution=0.1*np.pi, circle=False):
        """ :return pixel positions of ellipse points"""
        center = [r//2 for r in self.resolution]
        radius = [(r - c)*scale for r, c in zip(self.resolution, center)]

        alpha = np.arange(0, 2*np.pi, resolution)

        if circle:
            radius = [min(radius)]*2  # pick smaller radius to make a circle

        x = center[0] + radius[0]*np.cos(alpha)
        y = center[1] + radius[1]*np.sin(alpha)

        return x, y

        # DONE: calculate points of a circle (goniometry)
        # DONE: make function for pixel->angle / angle->pixel position conversion


class Servo:
    """

    """
    __angle_bounds = (0, 180)  # degrees
    __speed = 10  # deg/tick
    __slippage = 0  # deg (UNIMPLEMENTED)

    _default_angle = 90  # degrees

    def __init__(self):
        self._angle = self._default_angle
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
        return self.__angle_bounds, self._default_angle, self.__speed, self.__slippage

    def get_angle(self):
        return self._angle

    def get_default_angle(self):
        return self._default_angle

    @__enforce_bounds
    def set_angle(self, angle):
        self._angle = angle

    @__enforce_bounds
    def set_default_angle(self, angle):
        self._default_angle = angle

    @__enforce_bounds
    def move_to(self, angle):
        if angle == self._angle:
            pass
        elif angle < self._angle:
            self._angle = max(self._angle - self.__speed, angle)
        else:
            self._angle = min(self._angle + self.__speed, angle)

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
    def __init__(self, env=Environment(), servos=(None, None)):
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


class Construct:
    """

    """
    __vertical_laser_distance = 0.2  # m

    def __init__(self, lasers=(Laser(), Laser()), visualize=False):
        self._laser_red = lasers[0]
        self._laser_green = lasers[1]

        self.visualize = visualize

        if self.visualize:
            self.wall = Wall(blit=True)

    def run(self):
        pass

        # __ for each tick __

        # move red laser based on some pattern/path generator

        # move green laser based on agent decision

        # draw wall one tick (if visualize==True)


class Wall:
    """

    """

    def __init__(self, resolution=Environment.camera_resolution, blit=False):
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
        self.stm_red.markerline.set_data(red_pos[0], red_pos[1])
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

