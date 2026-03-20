import OpenGL.GL as gl
import numpy as np

class Shader:
    def __init__(self, vertex_str, fragment_str):
        
        vertex_shader = self._compile_shader(gl.GL_VERTEX_SHADER, vertex_str)
        fragment_shader = self._compile_shader(gl.GL_FRAGMENT_SHADER, fragment_str)
        self.program = self._link_program(vertex_shader, fragment_shader)
        gl.glDeleteShader(vertex_shader)
        gl.glDeleteShader(fragment_shader)
        
        self._uniform_locs = {}
    
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
    
    def set_uniform(self, name, value):
        
        loc = self.get_uniform_location(name)
        
        value = np.ascontiguousarray(value)
        if value.ndim != 1 or value.shape[0] > 4:
            raise ValueError(f"Unsupported uniform value shape: {value.shape}")
        
        if np.issubdtype(value.dtype, np.floating):
            value = value.astype(np.float32)
            func = [gl.glUniform1f, gl.glUniform2f, gl.glUniform3f, gl.glUniform4f][value.shape[0] - 1]
        elif np.issubdtype(value.dtype, np.signedinteger):
            value = value.astype(np.int32)
            func = [gl.glUniform1i, gl.glUniform2i, gl.glUniform3i, gl.glUniform4i][value.shape[0] - 1]
        elif np.issubdtype(value.dtype, np.unsignedinteger):
            value = value.astype(np.uint32)
            func = [gl.glUniform1ui, gl.glUniform2ui, gl.glUniform3ui, gl.glUniform4ui][value.shape[0] - 1]
        else:
            raise TypeError(f"Unsupported uniform type: {value.dtype}")
        
        with self:
            func(loc, *value)
    
    def set_uniform_array(self, name, value):
        
        loc = self.get_uniform_location(name)
        
        value = np.ascontiguousarray(value)
        if value.ndim != 2 or value.shape[1] > 4:
            raise ValueError(f"Unsupported uniform array shape: {value.shape}")
        
        if np.issubdtype(value.dtype, np.floating):
            value = value.astype(np.float32)
            func = [gl.glUniform1fv, gl.glUniform2fv, gl.glUniform3fv, gl.glUniform4fv][value.shape[1] - 1]
        elif np.issubdtype(value.dtype, np.signedinteger):
            value = value.astype(np.int32)
            func = [gl.glUniform1iv, gl.glUniform2iv, gl.glUniform3iv, gl.glUniform4iv][value.shape[1] - 1]
        elif np.issubdtype(value.dtype, np.unsignedinteger):
            value = value.astype(np.uint32)
            func = [gl.glUniform1uiv, gl.glUniform2uiv, gl.glUniform3uiv, gl.glUniform4uiv][value.shape[1] - 1]
        else:
            raise TypeError(f"Unsupported uniform array type: {value.dtype}")
        
        with self:
            func(loc, value.shape[0], value)
    
    def set_uniform_matrix(self, name, value, transpose=False):
        
        loc = self.get_uniform_location(name)
        
        value = np.ascontiguousarray(value)
        if value.ndim == 2:
            value = value[np.newaxis, ...]
        
        if value.ndim != 3 or \
            value.shape[1] < 2 or value.shape[1] > 4 or \
            value.shape[2] < 2 or value.shape[2] > 4:
            raise ValueError(f"Unsupported uniform matrix shape: {value.shape}")
        
        if np.issubdtype(value.dtype, np.floating):
            value = value.astype(np.float32)
            
            func_map = {
                (2, 2): gl.glUniformMatrix2fv,
                (2, 3): gl.glUniformMatrix2x3fv,
                (2, 4): gl.glUniformMatrix2x4fv,
                (3, 2): gl.glUniformMatrix3x2fv,
                (3, 3): gl.glUniformMatrix3fv,
                (3, 4): gl.glUniformMatrix3x4fv,
                (4, 2): gl.glUniformMatrix4x2fv,
                (4, 3): gl.glUniformMatrix4x3fv,
                (4, 4): gl.glUniformMatrix4fv
            }
            
            func = func_map.get((value.shape[1], value.shape[2]))
            if func is None:
                raise ValueError(f"Unsupported uniform matrix shape: {value.shape}")
        else:
            raise TypeError(f"Unsupported uniform matrix type: {value.dtype}")

        value = np.ascontiguousarray(value)
        with self:
            func(loc, value.shape[0], transpose, value)

    def get_uniform_location(self, name):
        
        if name not in self._uniform_locs:
            loc = gl.glGetUniformLocation(self.program, name)
            if loc == -1:
                raise ValueError(f"Uniform '{name}' not found in shader")
            self._uniform_locs[name] = loc
        
        return self._uniform_locs[name]

    def use(self):
        gl.glUseProgram(self.program)
    
    def stop(self):
        gl.glUseProgram(0)
    
    def __enter__(self):
        self.use()
        return self
    
    def __exit__(self, type_, value, traceback):
        self.stop()
