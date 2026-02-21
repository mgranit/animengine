import numpy as np
import matplotlib.colors as colors

def get_rgb_array_by_hsv(h_min, h_max, s, v, num):
    h_array = np.linspace(h_min, h_max, num)
    hsv_array = np.stack([h_array, np.full_like(h_array, s), np.full_like(h_array, v)], axis=1)
    rgb_array = colors.hsv_to_rgb(hsv_array)
    return rgb_array
