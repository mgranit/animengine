import numpy as np

def get_perspective_projection_matrix(fov_deg, aspect_ratio, near_plane, far_plane, dtype=np.float64):
    fov_rad = np.deg2rad(fov_deg)
    f = 1.0 / np.tan(fov_rad / 2.0)
    a = aspect_ratio
    n = near_plane
    f_p = far_plane

    return np.array([
        [f/a, 0.0, 0.0, 0.0],
        [0.0,   f, 0.0, 0.0],
        [0.0, 0.0, (f_p + n) / (n - f_p), (2 * f_p * n) / (n - f_p)],
        [0.0, 0.0, -1.0, 0.0]
    ], dtype=dtype)