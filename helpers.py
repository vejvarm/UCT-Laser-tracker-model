import numpy as np

from transformations import PathGenerator


def generate_path(path_type, **kwargs):
    initial_angles = tuple(kwargs["initial_angles"]) if "initial_angles" in kwargs.keys() else None
    angle_bounds = tuple(kwargs["angle_bounds"]) if "angle_bounds" in kwargs.keys() else (0, 180)
    max_angle_step = int(kwargs["max_angle_step"]) if "max_angle_step" in kwargs.keys() else 10
    scale = kwargs["scale"] if "scale" in kwargs.keys() else 0.5
    resolution = kwargs["resolution"] if "resolution" in kwargs.keys() else 0.1*np.pi

    path_gen = PathGenerator()

    if initial_angles:
        path_gen.initial_angles(initial_angles)

    if path_type == "random":
        xy = path_gen.random_gen(angle_bounds, max_angle_step, return_angles=True)
    elif path_type == "circle":
        xy = path_gen.ellipse(scale=scale, resolution=resolution, circle=True, return_angles=True)
    elif path_type == "ellipse":
        xy = path_gen.ellipse(scale=scale, resolution=resolution, circle=False, return_angles=True)
    else:
        raise NotImplementedError("Chosen path type doesn't exist/is not implemented yet.")

    return xy