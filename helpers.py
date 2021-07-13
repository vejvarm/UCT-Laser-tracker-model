import numpy as np

from environment import PathGenerator


def generate_path(path_type="circle", **kwargs):
    path_gen = PathGenerator()
    if "scale" in kwargs.keys():
        scale = kwargs["scale"]
    else:
        scale = 0.9
    if "resolution" in kwargs.keys():
        resolution = kwargs["resolution"]
    else:
        resolution = 0.1

    if path_type == "circle":
        x, y = path_gen.ellipse(scale=scale, resolution=resolution * np.pi, circle=True, return_angles=True)
    elif path_type == "ellipse":
        x, y = path_gen.ellipse(scale=scale, resolution=resolution * np.pi, circle=False, return_angles=True)
    else:
        raise NotImplementedError("Chosen path type doesn't exist/is not implemented yet.")

    return x, y