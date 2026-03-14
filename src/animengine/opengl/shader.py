import OpenGL.GL as gl

class Shader:
    def __init__(self, vertex_str, fragment_str):
        
        vertex_shader = self._compile_shader(gl.GL_VERTEX_SHADER, vertex_str)
        fragment_shader = self._compile_shader(gl.GL_FRAGMENT_SHADER, fragment_str)
        self.program = self._link_program(vertex_shader, fragment_shader)
        gl.glDeleteShader(vertex_shader)
        gl.glDeleteShader(fragment_shader)
    
    def _compile_shader(self, shader_type, source):
        
        shader = gl.glCreateShader(shader_type)
        gl.glShaderSource(shader, source)
        gl.glCompileShader(shader)
        if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
            info_log = gl.glGetShaderInfoLog(shader)
            gl.glDeleteShader(shader)
            raise RuntimeError(f"Shader {shader_type} compilation failed:\n{info_log.decode()}")
        
        return shader
    
    def _link_program(self, vertex_shader, fragment_shader):
        
        program = gl.glCreateProgram()
        gl.glAttachShader(program, vertex_shader)
        gl.glAttachShader(program, fragment_shader)
        gl.glLinkProgram(program)
        if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
            info_log = gl.glGetProgramInfoLog(program)
            gl.glDeleteProgram(program)
            raise RuntimeError(f"Shader program linking failed:\n{info_log.decode()}")
        
        return program
    
    def __enter__(self):
        gl.glUseProgram(self.program)
        return self
    
    def __exit__(self, type_, value, traceback):
        gl.glUseProgram(0)
    
    def __del__(self):
        if hasattr(self, 'program') and self.program:
            gl.glDeleteProgram(self.program)
