import numpy as np


def norm(v: np.ndarray):
    """
    Get norm 2 of given vector
    Args:
        v (np.ndarray): a vector

    Returns:
        (np.float) norm2 of vector
    """
    return np.linalg.norm(v)


def angular_between_two_vector(
        u: np.ndarray,
        v: np.ndarray
) -> float:
    """
    Return angular between 2 vectors in radian
    Args:
        u (np.ndarray): vector u
        v (np.ndarray): vector v

    Returns:
        (float) angular in radian
    """
    dot_product = np.dot(u, v)
    norm_u = norm(u)
    norm_v = norm(v)
    return np.arccos(dot_product / (norm_u * norm_v))


def vector3d_to_numpy(v):
    return np.array([v.x, v.y, v.z])
