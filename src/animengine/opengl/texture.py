import OpenGL.GL as gl
from PIL import Image
import numpy as np

class Texture:
    def __init__(self, path, target=gl.GL_TEXTURE_2D):
        self.target = target
        self.id = gl.glGenTextures(1)
        
        img = Image.open(path).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        img = img.convert("RGB")
        data = np.array(img, dtype=np.uint8)
        
        gl.glBindTexture(self.target, self.id)
            
        # Default parameters
        gl.glTexParameteri(self.target, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(self.target, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameteri(self.target, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(self.target, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        
        gl.glTexImage2D(
            self.target,
            0,
            gl.GL_RGB,
            img.width,
            img.height,
            0,
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            data
        )
        gl.glGenerateMipmap(self.target)
        
        gl.glBindTexture(self.target, 0)
        
        img.close()
        del data
    
    def use(self, unit):
        self._unit = unit
        return self

    def bind(self, unit=0):
        gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
        gl.glBindTexture(self.target, self.id)
    
    def unbind(self, unit=0):
        gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
        gl.glBindTexture(self.target, 0)
    
    def __enter__(self):
        self.bind(self._unit)
        return self

    def __exit__(self, *args):
        self.unbind(self._unit)
