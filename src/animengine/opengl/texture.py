import OpenGL.GL as gl
from PIL import Image
import numpy as np

def create_checkerboard(width=256, height=256, square_size=32):
        
    x = np.arange(width) // square_size
    y = np.arange(height) // square_size
    
    pattern = (x[np.newaxis, :] + y[:, np.newaxis]) % 2
    
    img_data = (pattern[:, :, np.newaxis] * 255).astype(np.uint8)
    img_data = np.repeat(img_data, 3, axis=2)
    
    return img_data

class Texture2D:
    def __init__(self, path: str | None = None, type_ = "diffuse"):
        self.type = type_
        self.id = gl.glGenTextures(1)
        
        if path is not None:
            img = Image.open(path).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            img = img.convert("RGB")
            data = np.array(img, dtype=np.uint8)
            img.close()
        else:
            data = create_checkerboard()
        
        with self:
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D,
                0,
                gl.GL_RGB,
                data.shape[1],
                data.shape[0],
                0,
                gl.GL_RGB,
                gl.GL_UNSIGNED_BYTE,
                data
            )
        
        del data

    def bind(self):
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.id)
    
    def unbind(self):
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    
    def __enter__(self):
        self.bind()
        return self

    def __exit__(self, *_):
        self.unbind()
