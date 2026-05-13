import numpy as np
import OpenGL.GL as gl

from animengine.opengl import Shader, VertexArray, Buffer
from animengine.transformations import get_translation_matrix

class CoordinateAxes:
    
    vertex_str = """
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aColor;

    out vec3 vertexColor;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main() {
        gl_Position = projection * view * model * vec4(aPos, 1.0);
        vertexColor = aColor;
    }
    """

    fragment_str = """
    #version 330 core
    in vec3 vertexColor;
    out vec4 FragColor;

    void main() {
        FragColor = vec4(vertexColor, 1.0);
    }
    """
    
    instances = []
    
    shader: Shader | None = None
    
    geometry_initialized: bool = False
    VBO: Buffer | None = None
    VAO: VertexArray | None = None
    
    grid_geometry_initialized: bool = False
    grid_VBO: Buffer | None = None
    grid_VAO: VertexArray | None = None
    n_grid_vertices: int = 0
    
    @classmethod
    def _init_geometry(cls) -> None:
        
        cls.shader = Shader(cls.vertex_str, cls.fragment_str)
        
        vertices = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],  # X axis
            [0.0, 0.0, 0.0], [0.0, 1.0, 0.0],  # Y axis
            [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],  # Z axis
        ], dtype=np.float32)
        
        colors = np.array([
            [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],  # X axis - red
            [0.0, 1.0, 0.0], [0.0, 1.0, 0.0],  # Y axis - green
            [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],  # Z axis - blue
        ], dtype=np.float32)
        
        data = np.hstack((vertices, colors))
        
        cls.VBO = Buffer(gl.GL_ARRAY_BUFFER, data=data)
        cls.VAO = VertexArray()
        cls.VAO.add_attribute(cls.VBO, 0, 3, gl.GL_FLOAT, 6 * data.itemsize, 0 * data.itemsize)
        cls.VAO.add_attribute(cls.VBO, 1, 3, gl.GL_FLOAT, 6 * data.itemsize, 3 * data.itemsize)
        
        cls.geometry_initialized = True
    
    @classmethod
    def _init_grid_geometry(cls) -> None:
        # Grid on XY plane
        grid_size = 10.0
        grid_step = 1.0
        grid_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)  # Gray color
        grid_vertices = []

        # Vertical lines (parallel to Y axis)
        for x in np.arange(-grid_size/2, grid_size/2 + grid_step, grid_step):
            grid_vertices.append([x, -grid_size/2, 0.0, *grid_color])
            grid_vertices.append([x, grid_size/2, 0.0, *grid_color])

        # Horizontal lines (parallel to X axis)
        for y in np.arange(-grid_size/2, grid_size/2 + grid_step, grid_step):
            grid_vertices.append([-grid_size/2, y, 0.0, *grid_color])
            grid_vertices.append([grid_size/2, y, 0.0, *grid_color])

        grid_vertices = np.array(grid_vertices, dtype=np.float32)

        cls.grid_VBO = Buffer(gl.GL_ARRAY_BUFFER, data=grid_vertices)
        cls.grid_VAO = VertexArray()
        cls.grid_VAO.add_attribute(cls.grid_VBO, 0, 3, gl.GL_FLOAT, 6 * grid_vertices.itemsize, 0 * grid_vertices.itemsize)
        cls.grid_VAO.add_attribute(cls.grid_VBO, 1, 3, gl.GL_FLOAT, 6 * grid_vertices.itemsize, 3 * grid_vertices.itemsize)
        
        cls.n_grid_vertices = len(grid_vertices)
        cls.grid_geometry_initialized = True
        
    def __init__(self, pos: list | np.ndarray | None = None, grid: bool = False) -> None:
        if not self.__class__.geometry_initialized:
            self.__class__._init_geometry()
        
        self.pos = np.array(pos if pos is not None else [0.0, 0.0, 0.0], dtype=np.float32)
        
        self.model_matrix = get_translation_matrix(*self.pos, dtype=np.float32)
        
        self.grid = grid
        if self.grid and not self.__class__.grid_geometry_initialized:
            self.__class__._init_grid_geometry()
        
        self.__class__.instances.append(self)
    
    def render(self, camera) -> None:
        shader = self.__class__.shader
        VAO = self.__class__.VAO
        
        assert shader is not None and VAO is not None, "CoordinateAxes geometry not initialized"
        
        with shader:
            shader.set_uniform_matrix("view", camera.view, True)
            shader.set_uniform_matrix("projection", camera.projection_matrix, True)
            
            shader.set_uniform_matrix("model", self.model_matrix, True)
            
            gl.glLineWidth(2.0)
            with VAO:
                gl.glDrawArrays(gl.GL_LINES, 0, 6)
            gl.glLineWidth(1.0)
            
            if self.grid:
                grid_VAO = self.__class__.grid_VAO
                assert grid_VAO is not None, "Grid geometry not initialized"
                
                gl.glLineWidth(1.0)
                with grid_VAO:
                    gl.glDrawArrays(gl.GL_LINES, 0, self.__class__.n_grid_vertices)
                gl.glLineWidth(1.0)
    
    @classmethod
    def render_all(cls, camera) -> None:
        shader = cls.shader
        VAO = cls.VAO

        assert shader is not None and VAO is not None, "CoordinateAxes geometry not initialized"
        
        with shader:
            shader.set_uniform_matrix("view", camera.view, True)
            shader.set_uniform_matrix("projection", camera.projection_matrix, True)
            
            for instance in cls.instances:
                shader.set_uniform_matrix("model", instance.model_matrix, True)
            
                gl.glLineWidth(2.0)
                with VAO:
                    gl.glDrawArrays(gl.GL_LINES, 0, 6)
                gl.glLineWidth(1.0)

                if instance.grid:
                    grid_VAO = cls.grid_VAO
                    assert grid_VAO is not None, "Grid geometry not initialized"
                    
                    gl.glLineWidth(1.0)
                    with grid_VAO:
                        gl.glDrawArrays(gl.GL_LINES, 0, cls.n_grid_vertices)
                    gl.glLineWidth(1.0)
