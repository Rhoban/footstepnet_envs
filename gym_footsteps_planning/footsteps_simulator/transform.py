import numpy as np


def rotation(angle: float = 0) -> np.ndarray:
    """
    A 2D homogeneous rotation (3x3 matrix)
    """
    # fmt: off
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]], dtype=np.float32)
    # fmt: on


def translation(x: float = 0, y: float = 0) -> np.ndarray:
    """
    A 2D homogeneous translation (3x3 matrix)
    """
    # fmt: off
    return np.array([[1, 0, x],
                      [0, 1, y],
                      [0, 0, 1]], dtype=np.float32)
    # fmt: on


def frame(x: float = 0, y: float = 0, angle: float = 0) -> np.ndarray:
    """
    A 2D transformation (rotation and translation, 3x3 matrix)
    """
    return translation(x, y) @ rotation(angle)


def frame_inv(T: np.ndarray) -> np.ndarray:
    """
    Inverts a 3x3 2D matrix
    """
    R = T[:2, :2]  # Rotation
    t = T[:2, 2:]  # Translation
    upper = np.hstack((R.T, -R.T @ t))
    lower = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return np.vstack((upper, lower))


def apply(T: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Apply a matrix transformation to a point
    """
    return (T @ [*point, 1.0])[:2]
