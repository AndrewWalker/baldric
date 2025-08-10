import numpy as np


def rotation_matrix_2d(theta: float):
    cos_angle = np.cos(theta)
    sin_angle = np.sin(theta)
    A = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
    return A
