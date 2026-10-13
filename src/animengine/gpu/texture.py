import numpy as np
import OpenGL.GL as gl


def create_checkerboard(width=256, height=256, square_size=32) -> np.ndarray:
    x = np.arange(width) // square_size
    y = np.arange(height) // square_size
    pattern = ((x[np.newaxis, :] + y[:, np.newaxis]) % 2 * 255).astype(np.uint8)
    return np.repeat(pattern[:, :, np.newaxis], 3, axis=2)


class Texture2D:
    """A 2D texture from an HxWx3 or HxWx4 uint8 array (row 0 is the top of the image)."""

    def __init__(self, data: np.ndarray | None = None, *, mipmaps: bool = True) -> None:
        if data is None:
            data = create_checkerboard()
        data = np.ascontiguousarray(data, dtype=np.uint8)
        if data.ndim != 3 or data.shape[2] not in (3, 4):
            raise ValueError(f"Expected HxWx3 or HxWx4 data, got shape {data.shape}")

        height, width, channels = data.shape
        fmt = gl.GL_RGB if channels == 3 else gl.GL_RGBA
        internal = gl.GL_RGB8 if channels == 3 else gl.GL_RGBA8

        self.id = gl.glGenTextures(1)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        with self:
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, internal, width, height, 0, fmt,
                            gl.GL_UNSIGNED_BYTE, np.flipud(data))
            if mipmaps:
                gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
                min_filter = gl.GL_LINEAR_MIPMAP_LINEAR
            else:
                min_filter = gl.GL_LINEAR
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, min_filter)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

    def bind(self) -> None:
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.id)

    def unbind(self) -> None:
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def __enter__(self) -> "Texture2D":
        self.bind()
        return self

    def __exit__(self, *_) -> None:
        self.unbind()
