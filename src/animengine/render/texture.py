from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(eq=False)
class Texture:
    """CPU-side image (HxWx3 or HxWx4 uint8, row 0 on top); the Renderer uploads it when drawn."""

    data: np.ndarray
    mipmaps: bool = True

    def __post_init__(self) -> None:
        self.data = np.ascontiguousarray(self.data, dtype=np.uint8)
        if self.data.ndim != 3 or self.data.shape[2] not in (3, 4):
            raise ValueError(f"Expected HxWx3 or HxWx4 data, got shape {self.data.shape}")

    @classmethod
    def from_file(cls, path, *, mipmaps: bool = True) -> "Texture":
        with Image.open(path) as img:
            has_alpha = "A" in img.getbands()
            data = np.array(img.convert("RGBA" if has_alpha else "RGB"), dtype=np.uint8)
        return cls(data, mipmaps=mipmaps)
