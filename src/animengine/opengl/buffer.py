import OpenGL.GL as gl

class Buffer:
    def __init__(self, target, data=None, usage=gl.GL_STATIC_DRAW):
        
        self.id = gl.glGenBuffers(1)
        self.target = target
        self.usage = usage
        
        if data is not None:
            self.set_data(data, usage)
    
    def set_data(self, data, usage=None):
        if usage is not None:
            self.usage = usage
        with self:
            gl.glBufferData(self.target, data.nbytes, data, self.usage)
    
    def update_data(self, data, offset=0, orphan=False):
        with self:
            if orphan:
                gl.glBufferData(self.target, data.nbytes, None, self.usage)
            gl.glBufferSubData(self.target, offset, data.nbytes, data)
        
    def bind(self):
        gl.glBindBuffer(self.target, self.id)
    
    def unbind(self):
        gl.glBindBuffer(self.target, 0)
    
    def __enter__(self):
        self.bind()
        return self
    
    def __exit__(self, type_, value, traceback):
        self.unbind()
