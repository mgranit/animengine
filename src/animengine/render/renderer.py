from collections.abc import Iterable
from importlib.resources import files
from typing import NamedTuple
from weakref import WeakKeyDictionary

import numpy as np
import OpenGL.GL as gl

from animengine.gpu import Buffer, Framebuffer, Shader, Texture2D, VertexArray

from .camera import Camera
from .color import to_rgba
from .material import DEFAULT_LINE_WIDTH, DEFAULT_POINT_SIZE, Material
from .mesh import DrawMode, Mesh
from .texture import Texture

_GL_MODES = {
    DrawMode.POINTS: gl.GL_POINTS,
    DrawMode.LINES: gl.GL_LINES,
    DrawMode.LINE_STRIP: gl.GL_LINE_STRIP,
    DrawMode.TRIANGLES: gl.GL_TRIANGLES,
}

# Attribute name -> shader input location.
_ATTRIBUTE_LOCATIONS = {"positions": 0, "colors": 1, "uvs": 2}

_REQUIRED_ATTRIBUTES = {"vertex_color": "colors", "textured": "uvs"}


class DrawItem(NamedTuple):
    mesh: Mesh
    material: Material
    model: np.ndarray


class _GpuMesh:
    def __init__(self, instanced: bool = False) -> None:
        self.vao = VertexArray()
        self.buffers: dict[str, Buffer] = {}
        self.index_buffer: Buffer | None = None
        self.uploaded: dict[str, int] = {}
        # Instanced meshes advance their vertex attributes once per instance, not per vertex.
        self.divisor = 1 if instanced else 0

    def sync(self, mesh: Mesh) -> None:
        usage = gl.GL_DYNAMIC_DRAW if mesh.dynamic else gl.GL_STATIC_DRAW
        for name in mesh.changed_since(self.uploaded):
            data = getattr(mesh, name)
            self.uploaded[name] = mesh.version(name)
            if name == "indices":
                # The element buffer binding is VAO state, so upload while the VAO is bound.
                if self.index_buffer is None:
                    self.index_buffer = Buffer(gl.GL_ELEMENT_ARRAY_BUFFER)
                with self.vao:
                    self.index_buffer.bind()
                    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, data.nbytes, data, usage)
            elif name in self.buffers:
                self.buffers[name].set_data(data, usage)
            else:
                buffer = Buffer(gl.GL_ARRAY_BUFFER, data, usage)
                self.buffers[name] = buffer
                self.vao.add_attribute(buffer, _ATTRIBUTE_LOCATIONS[name], data.shape[1],
                                       gl.GL_FLOAT, divisor=self.divisor)


class Renderer:
    """Draws DrawItems into a Framebuffer or the default framebuffer.

    Owns every GPU resource it creates (shaders and meshes are created lazily on first use),
    so it must be used with the OpenGL context that was current when it first drew.
    """

    def __init__(self) -> None:
        self._shaders: dict[str, Shader] = {}
        self._meshes: WeakKeyDictionary[Mesh, _GpuMesh] = WeakKeyDictionary()
        self._instanced_meshes: WeakKeyDictionary[Mesh, _GpuMesh] = WeakKeyDictionary()
        self._textures: WeakKeyDictionary[Texture, Texture2D] = WeakKeyDictionary()

    def _gpu_texture(self, texture: Texture) -> Texture2D:
        gpu = self._textures.get(texture)
        if gpu is None:
            gpu = Texture2D(texture.data, mipmaps=texture.mipmaps)
            self._textures[texture] = gpu
        return gpu

    def _shader(self, kind: str) -> Shader:
        shader = self._shaders.get(kind)
        if shader is None:
            directory = files("animengine.render") / "shaders"
            shader = Shader(
                (directory / f"{kind}.vert").read_text(),
                (directory / f"{kind}.frag").read_text(),
            )
            self._shaders[kind] = shader
        return shader

    def _gpu_mesh(self, mesh: Mesh, instanced: bool = False) -> _GpuMesh:
        cache = self._instanced_meshes if instanced else self._meshes
        gpu = cache.get(mesh)
        if gpu is None:
            gpu = _GpuMesh(instanced)
            cache[mesh] = gpu
        gpu.sync(mesh)
        return gpu

    def render(
        self,
        items: Iterable[DrawItem],
        camera: Camera,
        *,
        background="#000000",
        bar_color=None,
        target: Framebuffer | None = None,
        size: tuple[int, int] | None = None,
    ) -> None:
        if target is not None:
            target.bind()
            width, height = target.width, target.height
        else:
            if size is None:
                raise ValueError("size is required when rendering to the default framebuffer")
            width, height = size
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        x, y, viewport_width, viewport_height = camera.viewport_rect((width, height))
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        has_bars = (x, y, viewport_width, viewport_height) != (0, 0, width, height)
        clear_color = bar_color if has_bars and bar_color is not None else background
        gl.glDisable(gl.GL_SCISSOR_TEST)
        gl.glViewport(0, 0, width, height)
        gl.glClearColor(*to_rgba(clear_color))
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)  # type: ignore

        if has_bars:
            gl.glEnable(gl.GL_SCISSOR_TEST)
            gl.glScissor(x, y, viewport_width, viewport_height)
            gl.glClearColor(*to_rgba(background))
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)  # type: ignore
            gl.glDisable(gl.GL_SCISSOR_TEST)

        gl.glViewport(x, y, viewport_width, viewport_height)
        projection_view = camera.projection_matrix(
            viewport_width / viewport_height
        ) @ camera.view_matrix
        for item in items:
            self._draw(item, projection_view)

    def _draw(self, item: DrawItem, projection_view: np.ndarray) -> None:
        mesh, material = item.mesh, item.material
        if mesh.vertex_count == 0:
            return

        draws_points = mesh.mode is DrawMode.POINTS
        draws_lines = mesh.mode in (DrawMode.LINES, DrawMode.LINE_STRIP)
        
        required = _REQUIRED_ATTRIBUTES.get(material.kind)
        if required is not None and getattr(mesh, required) is None:
            raise ValueError(f"Material '{material.kind}' needs mesh.{required}")

        shader = self._shader(material.kind)
        instanced = material.kind == "glow"
        gpu = self._gpu_mesh(mesh, instanced)
        additive = material.blend == "additive"

        with shader:
            shader.set_uniform_matrix("mvp", projection_view @ item.model)
            if material.kind != "vertex_color":
                shader.set_uniform("color", material.color)
            if instanced:
                shader.set_uniform("extent", material.glow_extent())
                shader.set_uniform("core", material.glow_core)
                shader.set_uniform("half_distance", material.glow_half_distance)
                shader.set_uniform("power", material.glow_power)
                if mesh.colors is None:
                    gl.glVertexAttrib3f(_ATTRIBUTE_LOCATIONS["colors"], 1.0, 1.0, 1.0)
            gpu_texture = None
            if material.kind == "textured":
                if material.texture is None:
                    raise ValueError("Material 'textured' needs a texture")
                gpu_texture = self._gpu_texture(material.texture)
                gl.glActiveTexture(gl.GL_TEXTURE0)
                gpu_texture.bind()
                shader.set_uniform("tex", 0)

            if draws_points and material.point_size != DEFAULT_POINT_SIZE:
                gl.glPointSize(material.point_size)
            if draws_lines and material.line_width != DEFAULT_LINE_WIDTH:
                gl.glLineWidth(material.line_width)
            if additive:
                # Additive surfaces are order independent, so they must not hide each other.
                gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)
                gl.glDepthMask(gl.GL_FALSE)

            with gpu.vao:
                if instanced:
                    gl.glDrawArraysInstanced(gl.GL_TRIANGLE_STRIP, 0, 4, mesh.vertex_count)
                else:
                    mode = _GL_MODES[mesh.mode]
                    if mesh.indices is not None:
                        gl.glDrawElements(mode, len(mesh.indices), gl.GL_UNSIGNED_INT, None)
                    else:
                        gl.glDrawArrays(mode, 0, mesh.vertex_count)

            if draws_points and material.point_size != DEFAULT_POINT_SIZE:
                gl.glPointSize(DEFAULT_POINT_SIZE)
            if draws_lines and material.line_width != DEFAULT_LINE_WIDTH:
                gl.glLineWidth(DEFAULT_LINE_WIDTH)
            if additive:
                gl.glDepthMask(gl.GL_TRUE)
                gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            
            if gpu_texture is not None:
                gpu_texture.unbind()
