import itertools
from typing import Sequence

import tensorflow as tf
import numpy as np


class Environment:
    """

    """
    camera_resolution = (4056, 3040)  # (width, height) px
    camera_to_wall_distance = 4.  # m
    _chip_size = (7.564, 5.476)  # (width, height) mm
    _crop_factor = 5.54
    _lens_focal_length = (2.8, 12)  # mm
    _wall_diagonal_pixels = tf.sqrt(tf.reduce_sum(tf.square(tf.cast(camera_resolution, tf.float32))))

    def __init__(self):
        self.default_aov, self.default_fov = self.get_fov()
        self.default_ppm = (self.get_ppm(self.default_fov[0], 0), self.get_ppm(self.default_fov[1], 1))

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
        :param axis: (int) axis for which ppm is calculated | Horizontal (0) | Vertical (1) |
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
        return self.camera_resolution[axis]/2 + ppm * pos_meters

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
        alpha = (angle_defaults[0] - angle)/180*np.pi  # converted to rad
        beta = (angle_defaults[1] - angle2)/180*np.pi  # converted to rad
        return wall_dist*tf.tan(alpha)/tf.cos(beta)

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

    @tf.function
    def angle_to_pixel(self, angles: Sequence[int], angle_defaults=(90, 90), ppm=(None, None)):
        """
        :param angles: Tuple[int] x and y angles of the servos for respective laser
        :param angle_defaults: Tuple[int] default values for x and y angles of the servos

        :return: Tuple[int]
        """
        assert len(angles) == 2, "angles must hold exactly 2 values"
        assert len(angle_defaults) == 2, "angle defaults must hold exactly 2 values"

#        _, fov = self.get_fov()
        ppm = list(ppm)
        if ppm[0] is None:
            ppm[0] = self.default_ppm[0]
        if ppm[1] is None:
            ppm[1] = self.default_ppm[1]
#        ppm_x = self.get_ppm(fov[0], axis=0)
#        ppm_y = self.get_ppm(fov[1], axis=1)

        m_x = self.angle_to_meter(angles[0], angles[1], angle_defaults)
        m_y = self.angle_to_meter(angles[1], angles[0], angle_defaults[::-1])

        pixel_x = self.meter_to_pixel(m_x, ppm[0], axis=0)
        pixel_y = self.meter_to_pixel(m_y, ppm[1], axis=1)

        return pixel_x, pixel_y

    def cost(self, pred_batch, target_batch):
        """ euclidean distance cost function """
        # print(f"red: {pred_batch}")
        # print(f"grn: {target_batch}")
        return (tf.sqrt(tf.square(pred_batch[:, 0] - target_batch[:, 0])
                        + tf.square(pred_batch[:, 1] - target_batch[:, 1]))
                / self._wall_diagonal_pixels)

    def reward(self, red_batch, grn_batch):
        return 1. - self.cost(red_batch, grn_batch)


class PathGenerator:

    def __init__(self, resolution=Environment.camera_resolution):
        self.env = Environment()
        self.resolution = resolution

    def ellipse(self, scale=0.5, resolution=0.1*np.pi, circle=False, return_angles=False):
        """ generate pixel or servo angle points, which result in elliptical shape

        :param scale: (float) [0 to 1] size of the shape with respect to the size of working area
        :param resolution: (float) period (distance) of points in the shape
        :param circle: (bool) if True, calculate shape to be a circle | else it's an ellipse
        :param return_angles: if True, return angles of servos | else return pixel positions
        :return pixel positions/angles of ellipse/circle points
        """
        center = [r//2 for r in self.resolution]
        radius = [(r - c)*scale for r, c in zip(self.resolution, center)]

        alpha = np.arange(0, 2*np.pi, resolution)

        if circle:
            radius = [min(radius)]*2  # pick smaller radius to make a circle

        x = center[0] + radius[0]*np.cos(alpha)
        y = center[1] + radius[1]*np.sin(alpha)

        if return_angles:
            x, y = self._generate_ellipse_angles(x, y)

        return itertools.cycle(x), itertools.cycle(y)

    def _generate_ellipse_angles(self, xs, ys):
        # convert pixel path to servo angles
        aov, fov = self.env.get_fov()
        ppm_x = self.env.get_ppm(fov[1], axis=1)
        ppm_y = self.env.get_ppm(fov[0], axis=0)

        xas, yas = list(), list()
        ya = 90
        for x, y in zip(xs, ys):
            xm = self.env.pixel_to_meter(x, ppm_x, axis=0)
            ym = self.env.pixel_to_meter(y, ppm_y, axis=1)
            xa = self.env.meter_to_angle(xm, ya)
            ya = self.env.meter_to_angle(ym, xa)
            xas.append(xa)
            yas.append(ya)

        return xas, yas

        # DONE: calculate points of a circle (goniometry)
        # DONE: make function for pixel->angle / angle->pixel position conversion