import OpenGL.GL as GL
import numpy as np

OPENGL_TYPES = [
    'float', 'vec2', 'vec3', 'vec4',
    'int', 'ivec2', 'ivec3', 'ivec4',
    'uint', 'uvec2', 'uvec3', 'uvec4',
    'bool', 'bvec2', 'bvec3', 'bvec4',
]

# ---------------------------------------------------------------------------- #
# FUNCTIONS
# ---------------------------------------------------------------------------- #

def read_shader(file_path):
    with open(file_path, 'r') as file:
        shader_str = file.read()
    return shader_str

def compile_shader(shader_str, shader_type):
    
    # compile shader
    shader_ref = GL.glCreateShader(shader_type)
    GL.glShaderSource(shader_ref, shader_str)
    GL.glCompileShader(shader_ref)

    # check for errors
    if not GL.glGetShaderiv(shader_ref, GL.GL_COMPILE_STATUS):
        error_message = GL.glGetShaderInfoLog(shader_ref)
        GL.glDeleteShader(shader_ref)
        raise Exception(error_message.decode("utf-8"))
    
    return shader_ref

def initialize_program(vertex_str, fragment_str):
    
    # setup shaders
    vertex_ref = compile_shader(vertex_str, GL.GL_VERTEX_SHADER)
    fragment_ref = compile_shader(fragment_str, GL.GL_FRAGMENT_SHADER)
    
    # link program
    program_ref = GL.glCreateProgram()
    GL.glAttachShader(program_ref, vertex_ref)
    GL.glAttachShader(program_ref, fragment_ref)
    GL.glLinkProgram(program_ref)

    # check for errors
    if not GL.glGetProgramiv(program_ref, GL.GL_LINK_STATUS):
        error_message = GL.glGetProgramInfoLog(program_ref)
        GL.glDeleteProgram(program_ref)
        raise Exception(error_message.decode("utf-8"))

    return program_ref

# ---------------------------------------------------------------------------- #
# OBJECTS
# ---------------------------------------------------------------------------- #

class Shader:
    def __init__(self, vertex_str, fragment_str):
        
        self.vertex_str = vertex_str
        self.fragment_str = fragment_str
        
        self.uniforms = {}
        self.textures = {}
        
    def setup(self):
        
        self.program_ref = initialize_program(self.vertex_str, self.fragment_str)
        
        self.pos_data = np.array([
            [ 1.0,  1.0],
            [-1.0,  1.0],
            [-1.0, -1.0],
            [ 1.0, -1.0]
        ], dtype=np.float32)
        
        # create and bind VAO
        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)
        
        # create, bind, and fill VBO
        self.vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.pos_data.nbytes, self.pos_data.ravel(), GL.GL_STATIC_DRAW)
        
        # set up vertex attribute
        self.pos_attrib = GL.glGetAttribLocation(self.program_ref, "pos")
        GL.glVertexAttribPointer(self.pos_attrib, 2, GL.GL_FLOAT, False, 0, None)
        GL.glEnableVertexAttribArray(self.pos_attrib)
        
        # unbind
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)
    
    def draw(self, frame_num=None):
        
        GL.glUseProgram(self.program_ref)
        
        for texture in self.textures.values():
            texture.bind()
            texture.update()
        
        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, 4)
        GL.glBindVertexArray(0)
        
        GL.glUseProgram(0)

class Uniform:
    def __init__(self, shader, type_, name, data):
        
        self.shader = shader
        
        self.type_ = type_
        self.name = name
        self.is_array = False
        
        # check that type_ is a valid OpenGL uniform type
        if self.type_ not in OPENGL_TYPES:
            raise Exception(f"unsupported uniform type \'{self.type_}\'")
        
        # get uniform location
        self.loc = GL.glGetUniformLocation(self.shader.program_ref, self.name)
        if self.loc == -1:
            raise Exception(f"uniform '{self.name}' not found in program {self.shader.program_ref}")
        
        self.format_data(data)
        self.set_data(data, format_data=False)
        
        self.shader.uniforms[self.name] = self
    
    @property
    def gl_func(self):
        if self.is_array:
            func_name = f'glUniform{self.data.shape[1]}{self.type_suffix}v'
        else:
            func_name = f'glUniform{self.data.shape[0]}{self.type_suffix}'
        gl_func = getattr(GL, func_name, None)
        if gl_func is None:
            raise Exception(f"{func_name} not supported")
        return gl_func

    def format_data(self, data):
        
        if isinstance(data, bool):
            data = int(data)
        
        self.data = np.atleast_1d(data).copy()
        
        # check dimension of data
        if (self.data.ndim == 1 and self.data.shape[0] > 4) or (self.data.ndim == 2 and self.data.shape[1] > 4):
            raise Exception(f"uniform '{self.name}' data dimension must be 1-4, got {self.data.shape[1]}")
        
        # check that data is numeric
        if not np.issubdtype(self.data.dtype, np.number):
            raise Exception(f"uniform '{self.name}' data must be a numeric type, got {self.data.dtype}")
        
        # set data type and type suffix
        if self.type_ in ['float', 'vec2', 'vec3', 'vec4']:
            self.data = self.data.astype(np.float32)
            self.type_suffix = 'f'
        elif self.type_ in ['int', 'ivec2', 'ivec3', 'ivec4']:
            self.data = self.data.astype(np.int32)
            self.type_suffix = 'i'
        elif self.type_ in ['uint', 'uvec2', 'uvec3', 'uvec4']:
            self.data = self.data.astype(np.uint32)
            self.type_suffix = 'ui'
        elif self.type_ in ['bool', 'bvec2', 'bvec3', 'bvec4']:
            self.data = self.data.astype(np.int32)
            self.type_suffix = 'i'
        
        # check if uniform is array
        if self.data.ndim == 2:
            self.is_array = True
    
    def set_data(self, data, format_data=True):
        
        if format_data:
            self.format_data(data)
        
        GL.glUseProgram(self.shader.program_ref)
        if self.is_array:
            self.gl_func(self.loc, self.data.shape[0], self.data.ravel())
        else:
            self.gl_func(self.loc, *self.data)
        GL.glUseProgram(0)

class TextureData:
    def __init__(self, shader, data, internal_format, uniform_name):
        
        self.shader = shader
        
        self.data = data.copy().astype(np.float32)
        self.uniform_name = uniform_name
        
        # set up buffer
        self.buffer = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_TEXTURE_BUFFER, self.buffer)
        GL.glBufferData(GL.GL_TEXTURE_BUFFER, self.data.nbytes, self.data.ravel(), GL.GL_DYNAMIC_DRAW)
        GL.glBindBuffer(GL.GL_TEXTURE_BUFFER, 0)

        # set up texture
        self.texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_BUFFER, self.texture)
        GL.glTexBuffer(GL.GL_TEXTURE_BUFFER, internal_format, self.buffer)

        self.uniform_loc = GL.glGetUniformLocation(self.shader.program_ref, self.uniform_name)
        if self.uniform_loc == -1:
            print(f"Warning: uniform '{self.uniform_name}' not found in program {self.shader.program_ref}")
        
        self.shader.textures[self.uniform_name] = self
        self.texture_unit = len(self.shader.textures) - 1
    
    def bind(self):
        
        GL.glActiveTexture(int(GL.GL_TEXTURE0) + self.texture_unit)
        GL.glBindTexture(GL.GL_TEXTURE_BUFFER, self.texture)
        GL.glUniform1i(self.uniform_loc, self.texture_unit)
    
    def update(self):
        
        GL.glBindBuffer(GL.GL_TEXTURE_BUFFER, self.buffer)
        GL.glBufferSubData(GL.GL_TEXTURE_BUFFER, 0, self.data.nbytes, self.data.ravel())
        GL.glBindBuffer(GL.GL_TEXTURE_BUFFER, 0)
