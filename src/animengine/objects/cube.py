import numpy as np
import OpenGL.GL as gl

from animengine.opengl import Shader, VertexArray, Buffer, Texture2D
from animengine.camera import Camera
from animengine.transformations import get_translation_matrix, get_scaling_matrix

class Cube:
    
    vertex_str = """
    #version 330 core
    
    layout (location = 0) in vec3 aPos;
    
    uniform mat4 PVM;
    
    void main() {
        gl_Position = PVM * vec4(aPos, 1.0);
    }
    """
    
    fragment_str = """
    #version 330 core
    
    out vec4 FragColor;
    
    void main() {
        FragColor = vec4(1.0, 1.0, 1.0, 1.0);
    }
    """
    
    instances = []
    
    shader: Shader | None = None
    
    geometry_initialized: bool = False
    VBO: Buffer | None = None
    VAO: VertexArray | None = None
    
    @classmethod
    def _init_geometry(cls) -> None:
        
        cls.shader = Shader(cls.vertex_str, cls.fragment_str)
        
        vertices = np.array([
            # XY plane
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5],
            [ 0.5,  0.5, -0.5], [ 0.5, -0.5, -0.5], [-0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5],
            [ 0.5,  0.5,  0.5], [ 0.5, -0.5,  0.5], [-0.5, -0.5,  0.5],
            # YZ plane
            [-0.5, -0.5, -0.5], [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5],
            [-0.5,  0.5,  0.5], [-0.5,  0.5, -0.5], [-0.5, -0.5, -0.5],
            [ 0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5],
            [ 0.5,  0.5,  0.5], [ 0.5,  0.5, -0.5], [ 0.5, -0.5, -0.5],
            # ZX plane
            [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5],
            [ 0.5, -0.5,  0.5], [-0.5, -0.5,  0.5], [-0.5, -0.5, -0.5],
            [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5], [ 0.5,  0.5,  0.5],
            [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5], [-0.5,  0.5, -0.5],
        ], dtype=np.float32)
        
        cls.VBO = Buffer(gl.GL_ARRAY_BUFFER, data=vertices, usage=gl.GL_STATIC_DRAW)
        cls.VAO = VertexArray()
        cls.VAO.add_attribute(
            buffer=cls.VBO,
            index=0,
            size=3,
            dtype=gl.GL_FLOAT,
            stride=3 * vertices.itemsize,
            offset=0
        )
        cls.geometry_initialized = True
    
    def __init__(self, origin: list | np.ndarray | None = None, scale: float = 1.0) -> None:
        if not self.__class__.geometry_initialized:
            self.__class__._init_geometry()
        
        if origin is None:
            origin = [0.0, 0.0, 0.0]
        self.origin = np.array(origin, dtype=np.float32)
        self.scale = scale
        
        T = get_translation_matrix(*self.origin, dtype=np.float32)
        S = get_scaling_matrix(self.scale, self.scale, self.scale, dtype=np.float32)
        
        self.model_matrix = T @ S
        
        self.__class__.instances.append(self)
    
    def render(self, camera: Camera) -> None:
        shader = self.__class__.shader
        VAO = self.__class__.VAO
        
        assert shader is not None and VAO is not None, "Cube geometry not initialized"
        
        PVM = camera.projection_matrix @ camera.view @ self.model_matrix
        with shader:
            shader.set_uniform_matrix("PVM", PVM, True)
            
            with VAO:
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, 36)

class TexturedCube:
    
    vertex_str = """
    #version 330 core
    
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec2 aTexCoord;
    
    uniform mat4 PVM;
    
    out vec2 TexCoord;
    
    void main() {
        gl_Position = PVM * vec4(aPos, 1.0);
        TexCoord = aTexCoord;
    }
    """
    
    fragment_str = """
    #version 330 core
    
    in vec2 TexCoord;
    
    uniform sampler2D tex_sampler;
    
    out vec4 FragColor;
    
    void main() {
        FragColor = texture(tex_sampler, TexCoord);
    }
    """
    
    instances = []
    
    shader: Shader | None = None
    
    geometry_initialized: bool = False
    VBO: Buffer | None = None
    VAO: VertexArray | None = None
    
    @classmethod
    def _init_geometry(cls) -> None:
        
        cls.shader = Shader(cls.vertex_str, cls.fragment_str)
        
        vertices = np.array([
            # XY plane
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5],
            [ 0.5,  0.5, -0.5], [ 0.5, -0.5, -0.5], [-0.5, -0.5, -0.5],
            [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5],
            [ 0.5,  0.5,  0.5], [ 0.5, -0.5,  0.5], [-0.5, -0.5,  0.5],
            # YZ plane
            [-0.5, -0.5, -0.5], [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5],
            [-0.5,  0.5,  0.5], [-0.5,  0.5, -0.5], [-0.5, -0.5, -0.5],
            [ 0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5],
            [ 0.5,  0.5,  0.5], [ 0.5,  0.5, -0.5], [ 0.5, -0.5, -0.5],
            # ZX plane
            [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5],
            [ 0.5, -0.5,  0.5], [-0.5, -0.5,  0.5], [-0.5, -0.5, -0.5],
            [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5], [ 0.5,  0.5,  0.5],
            [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5], [-0.5,  0.5, -0.5],
        ], dtype=np.float32)
        
        tex_coords = np.array([
            # XY plane
            [0.0, 0.0], [0.0, 1.0], [1.0, 1.0],
            [1.0, 1.0], [1.0, 0.0], [0.0, 0.0],
            [0.0, 0.0], [0.0, 1.0], [1.0, 1.0],
            [1.0, 1.0], [1.0, 0.0], [0.0, 0.0],
            # YZ plane
            [0.0, 0.0], [0.0, 1.0], [1.0, 1.0],
            [1.0, 1.0], [1.0, 0.0], [0.0, 0.0],
            [0.0, 0.0], [0.0, 1.0], [1.0, 1.0],
            [1.0, 1.0], [1.0, 0.0], [0.0, 0.0],
            # ZX plane
            [0.0, 0.0], [0.0, 1.0], [1.0, 1.0],
            [1.0, 1.0], [1.0, 0.0], [0.0, 0.0],
            [0.0, 0.0], [0.0, 1.0], [1.0, 1.0],
            [1.0, 1.0], [1.0, 0.0], [0.0, 0.0],
        ], dtype=np.float32)
        
        cls.VBO = Buffer(gl.GL_ARRAY_BUFFER, data=np.hstack((vertices, tex_coords)), usage=gl.GL_STATIC_DRAW)
        cls.VAO = VertexArray()
        cls.VAO.add_attribute(
            buffer=cls.VBO,
            index=0,
            size=3,
            dtype=gl.GL_FLOAT,
            stride=5 * vertices.itemsize,
            offset=0 * vertices.itemsize
        )
        cls.VAO.add_attribute(
            buffer=cls.VBO,
            index=1,
            size=2,
            dtype=gl.GL_FLOAT,
            stride=5 * vertices.itemsize,
            offset=3 * vertices.itemsize
        )
        cls.geometry_initialized = True
        
    def __init__(
        self,
        origin: list | np.ndarray | None = None,
        scale: float = 1.0,
        texture_path: str | None = None
    ):
        if not self.__class__.geometry_initialized:
            self.__class__._init_geometry()
        
        if origin is None:
            origin = [0.0, 0.0, 0.0]
        self.origin = np.array(origin, dtype=np.float32)
        self.scale = scale
        
        T = get_translation_matrix(*self.origin, dtype=np.float32)
        S = get_scaling_matrix(self.scale, self.scale, self.scale, dtype=np.float32)
        
        self.model_matrix = T @ S
        
        self.texture = Texture2D(texture_path)
    
    def render(self, camera: Camera) -> None:
        shader = self.__class__.shader
        VAO = self.__class__.VAO
        
        assert shader is not None and VAO is not None, "Cube geometry not initialized"
        
        PVM = camera.projection_matrix @ camera.view @ self.model_matrix
        with shader:
            shader.set_uniform_matrix("PVM", PVM, True)
            
            with VAO:
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture.id)
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, 36)
                gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
