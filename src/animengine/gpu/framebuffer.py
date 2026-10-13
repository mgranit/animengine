import numpy as np
import OpenGL.GL as gl


def _make_renderbuffer(width: int, height: int, fmt: int, samples: int) -> int:
    rb = gl.glGenRenderbuffers(1)
    gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, rb)
    if samples > 0:
        gl.glRenderbufferStorageMultisample(gl.GL_RENDERBUFFER, samples, fmt, width, height)
    else:
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, fmt, width, height)
    gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, 0)
    return rb


class Framebuffer:
    """Offscreen render target with colour and depth, optionally multisampled."""

    def __init__(self, width: int, height: int, samples: int = 0) -> None:
        self.width = width
        self.height = height
        self.samples = samples

        # Single-sample target that holds the (resolved) colour image.
        self._resolve_fbo = gl.glGenFramebuffers(1)
        self.color_texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.color_texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, width, height, 0,
                        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._resolve_fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
                                  gl.GL_TEXTURE_2D, self.color_texture, 0)
        self._renderbuffers = []

        if samples > 0:
            self._render_fbo = gl.glGenFramebuffers(1)
            color_rb = _make_renderbuffer(width, height, gl.GL_RGBA8, samples)
            self._renderbuffers.append(color_rb)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._render_fbo)
            gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
                                         gl.GL_RENDERBUFFER, color_rb)
        else:
            self._render_fbo = self._resolve_fbo

        depth_rb = _make_renderbuffer(width, height, gl.GL_DEPTH_COMPONENT24, samples)
        self._renderbuffers.append(depth_rb)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT,
                                     gl.GL_RENDERBUFFER, depth_rb)

        for fbo in {self._resolve_fbo, self._render_fbo}:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
            status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
            if status != gl.GL_FRAMEBUFFER_COMPLETE:
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
                raise RuntimeError(f"Framebuffer incomplete (status 0x{status:x})")
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def bind(self) -> None:
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._render_fbo)
        gl.glViewport(0, 0, self.width, self.height)

    def unbind(self) -> None:
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def read_pixels(self) -> np.ndarray:
        """Return the image as a top-down HxWx3 uint8 array."""
        if self.samples > 0:
            gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self._render_fbo)
            gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self._resolve_fbo)
            gl.glBlitFramebuffer(0, 0, self.width, self.height, 0, 0, self.width, self.height,
                                 gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)

        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self._resolve_fbo)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        raw = gl.glReadPixels(0, 0, self.width, self.height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, 0)

        image = np.frombuffer(bytes(raw), dtype=np.uint8).reshape(self.height, self.width, 3)
        return np.ascontiguousarray(image[::-1])

    def __enter__(self) -> "Framebuffer":
        self.bind()
        return self

    def __exit__(self, *_) -> None:
        self.unbind()
