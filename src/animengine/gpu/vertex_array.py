import OpenGL.GL as gl
from ctypes import c_void_p

class VertexArray:
    def __init__(self):
        self.id = gl.glGenVertexArrays(1)
    
    def add_attribute(self, buffer, index, size, dtype, stride=0, offset=0, normalized=False,
                      divisor=0):
        if buffer.target != gl.GL_ARRAY_BUFFER:
            raise ValueError("Buffer target must be GL_ARRAY_BUFFER to be used as vertex attribute")
        
        with self, buffer:
            gl.glVertexAttribPointer(
                index,
                size,
                dtype,
                gl.GL_TRUE if normalized else gl.GL_FALSE,
                stride,
                c_void_p(offset)
            )
            gl.glEnableVertexAttribArray(index)
            if divisor:
                gl.glVertexAttribDivisor(index, divisor)
    
    def bind_element_buffer(self, buffer):
        if buffer.target != gl.GL_ELEMENT_ARRAY_BUFFER:
            raise ValueError(
                "Buffer target must be GL_ELEMENT_ARRAY_BUFFER to be used as element buffer"
            )
        
        with self:
            buffer.bind()
        buffer.unbind()
    
    def bind(self):
        gl.glBindVertexArray(self.id)
    
    def unbind(self):
        gl.glBindVertexArray(0)
    
    def enable_attribute(self, index):
        with self:
            gl.glEnableVertexAttribArray(index)
    
    def disable_attribute(self, index):
        with self:
            gl.glDisableVertexAttribArray(index)
    
    def __enter__(self):
        self.bind()
        return self
    
    def __exit__(self, *_):
        self.unbind()
