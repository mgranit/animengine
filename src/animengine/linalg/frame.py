import numpy as np

from .transforms import (
    get_rotation_matrix,
    get_rotation_matrix_x,
    get_rotation_matrix_y,
    get_rotation_matrix_z,
)

_TOLERANCE = 1e-6


class Frame:
    """A right-handed rigid coordinate frame: an origin and an orthonormal basis.

    `origin` and the columns of `basis` (the x, y and z axes) are expressed in the parent frame,
    so `matrix` maps local coordinates to parent coordinates. Frames are immutable, and scale is
    deliberately not representable.

    A frame does not know its parent: it is a plain pose value, and combining frames (`compose`,
    `relative_to`, `transform_point_to`) is only meaningful when they share the same parent.

    Repeatedly composing frames accumulates rounding error; call `orthonormalized` now and then
    on long-lived frames.
    """

    __slots__ = ("_origin", "_basis")

    def __init__(self, origin=(0.0, 0.0, 0.0), basis=None) -> None:
        origin = np.array(origin, dtype=np.float64)
        basis = np.eye(3) if basis is None else np.array(basis, dtype=np.float64)
        if origin.shape != (3,):
            raise ValueError(f"origin must have shape (3,), got {origin.shape}")
        if basis.shape != (3, 3):
            raise ValueError(f"basis must have shape (3, 3), got {basis.shape}")
        if not np.allclose(basis.T @ basis, np.eye(3), atol=_TOLERANCE):
            raise ValueError("basis must be orthonormal (use Frame.orthonormalized to repair)")
        if np.linalg.det(basis) <= 0.0:
            raise ValueError("basis must be right-handed")
        origin.setflags(write=False)
        basis.setflags(write=False)
        self._origin = origin
        self._basis = basis

    @property
    def origin(self) -> np.ndarray:
        return self._origin

    @property
    def basis(self) -> np.ndarray:
        return self._basis

    @property
    def x_axis(self) -> np.ndarray:
        return self._basis[:, 0]

    @property
    def y_axis(self) -> np.ndarray:
        return self._basis[:, 1]

    @property
    def z_axis(self) -> np.ndarray:
        return self._basis[:, 2]

    @classmethod
    def look_at(cls, origin, target, up) -> "Frame":
        """Camera-style frame: -z points from `origin` to `target`, +y is as close to `up` as
        possible and +x is to the right. If `up` is parallel to the view direction, the world
        axis least aligned with it is used instead.
        """
        origin = np.asarray(origin, dtype=np.float64)
        forward = np.asarray(target, dtype=np.float64) - origin
        length = np.linalg.norm(forward)
        if length == 0.0:
            raise ValueError("origin and target must differ")
        forward = forward / length

        up = np.asarray(up, dtype=np.float64)
        right = np.cross(forward, up)
        right_length = np.linalg.norm(right)
        if right_length < _TOLERANCE * max(np.linalg.norm(up), 1.0):
            fallback = np.eye(3)[np.argmin(np.abs(forward))]
            right = np.cross(forward, fallback)
            right_length = np.linalg.norm(right)
        right = right / right_length
        true_up = np.cross(right, forward)
        return cls(origin, np.column_stack((right, true_up, -forward)))

    @classmethod
    def from_axis_angle(cls, angle_deg: float, axis, origin=(0.0, 0.0, 0.0)) -> "Frame":
        return cls(origin, get_rotation_matrix(angle_deg, axis)[:3, :3])

    @classmethod
    def from_euler(cls, rotation_deg, origin=(0.0, 0.0, 0.0)) -> "Frame":
        """Rotation in degrees about x, y, z, applied in z-y-x order (as `Node.rotation`)."""
        rx, ry, rz = rotation_deg
        rotation = (
            get_rotation_matrix_z(rz) @ get_rotation_matrix_y(ry) @ get_rotation_matrix_x(rx)
        )
        return cls(origin, rotation[:3, :3])

    @classmethod
    def from_matrix(cls, matrix, *, orthonormalize: bool = False) -> "Frame":
        """Build from a 4x4 matrix.

        A rigid matrix is required unless `orthonormalize` is set, in which case the nearest
        rotation to its linear part is used and any scale, shear or reflection is discarded.
        """
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"matrix must have shape (4, 4), got {matrix.shape}")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=_TOLERANCE):
            raise ValueError("matrix must be affine")
        basis = matrix[:3, :3]
        if orthonormalize:
            u, _, vt = np.linalg.svd(basis)
            if np.linalg.det(u @ vt) < 0.0:
                u[:, -1] *= -1.0
            basis = u @ vt
        return cls(matrix[:3, 3], basis)

    def to_euler(self) -> np.ndarray:
        """Rotation in degrees about x, y, z in z-y-x order, the inverse of `from_euler`.

        At gimbal lock (y at +-90 degrees) the z rotation is reported as 0.
        """
        b = self._basis
        sin_y = -b[2, 0]
        if abs(sin_y) < 1.0 - 1e-12:
            ry = np.arcsin(sin_y)
            rx = np.arctan2(b[2, 1], b[2, 2])
            rz = np.arctan2(b[1, 0], b[0, 0])
        else:
            ry = np.copysign(np.pi / 2.0, sin_y)
            rx = np.arctan2(b[0, 1], b[0, 2]) if sin_y > 0 else np.arctan2(-b[0, 1], -b[0, 2])
            rz = 0.0
        return np.degrees([rx, ry, rz])

    def orthonormalized(self) -> "Frame":
        """The nearest rotation to this basis, to cancel accumulated rounding error."""
        u, _, vt = np.linalg.svd(self._basis)
        return Frame(self._origin, u @ vt)

    def matrix(self, dtype=np.float64) -> np.ndarray:
        """4x4 matrix taking local coordinates to parent coordinates."""
        result = np.eye(4)
        result[:3, :3] = self._basis
        result[:3, 3] = self._origin
        return result.astype(dtype)

    def inverse_matrix(self, dtype=np.float64) -> np.ndarray:
        """4x4 matrix taking parent coordinates to local coordinates."""
        result = np.eye(4)
        result[:3, :3] = self._basis.T
        result[:3, 3] = -self._basis.T @ self._origin
        return result.astype(dtype)

    def inverse(self) -> "Frame":
        return Frame(-self._basis.T @ self._origin, self._basis.T)

    def compose(self, other: "Frame") -> "Frame":
        """`other` is expressed in this frame; the result expresses it in this frame's parent."""
        return Frame(self._origin + self._basis @ other._origin, self._basis @ other._basis)

    def __matmul__(self, other: "Frame") -> "Frame":
        if not isinstance(other, Frame):
            return NotImplemented
        return self.compose(other)

    def transform_point(self, point) -> np.ndarray:
        """Local to parent coordinates; accepts shape (3,) or (N, 3)."""
        return np.asarray(point, dtype=np.float64) @ self._basis.T + self._origin

    def transform_vector(self, vector) -> np.ndarray:
        """Local to parent directions (no translation); accepts shape (3,) or (N, 3)."""
        return np.asarray(vector, dtype=np.float64) @ self._basis.T

    def to_local_point(self, point) -> np.ndarray:
        """Parent to local coordinates; accepts shape (3,) or (N, 3)."""
        return (np.asarray(point, dtype=np.float64) - self._origin) @ self._basis

    def to_local_vector(self, vector) -> np.ndarray:
        """Parent to local directions (no translation); accepts shape (3,) or (N, 3)."""
        return np.asarray(vector, dtype=np.float64) @ self._basis

    def relative_to(self, other: "Frame") -> "Frame":
        """This frame expressed in `other`; both must be expressed in the same parent frame."""
        return other.inverse().compose(self)

    def transform_point_to(self, other: "Frame", point) -> np.ndarray:
        """Point given in this frame's coordinates, expressed in `other`'s coordinates.

        Both frames must be expressed in the same parent frame.
        """
        return other.to_local_point(self.transform_point(point))

    def transform_vector_to(self, other: "Frame", vector) -> np.ndarray:
        """Direction given in this frame's coordinates, expressed in `other`'s coordinates.

        Both frames must be expressed in the same parent frame.
        """
        return other.to_local_vector(self.transform_vector(vector))

    def __repr__(self) -> str:
        return f"Frame(origin={self._origin.tolist()}, basis={self._basis.tolist()})"
