import numpy as np
from matplotlib import colors
from .sound import create_bounce_soundwave

class Star:
    def __init__(self, pos, vel, acc, col_hsv, lum, pow, frequency=400.0, volume=1.0):
        
        # kinematics
        self.pos = pos
        self.vel = vel
        self.acc = acc
        
        # visuals
        self.col_rgb = colors.hsv_to_rgb(col_hsv)
        self.lum_orig = lum
        self.lum = lum
        self.pow = pow
        
        # sound
        self.frequency = frequency
        self.volume = 1.0

        self.impulse_time = -np.inf
    
    def impulse(self, factor, time):
        self.lum *= factor
        self.impulse_time = time

    def update(self, dt, decay_rate=2.0):
        if self.lum > self.lum_orig:
            self.lum = self.lum_orig + (self.lum - self.lum_orig) * np.exp(-decay_rate * dt)

class StarGroup:
    def __init__(
        self,
        pos_init,
        shape,
        colors_hsv,
        frequencies,
        update_func,
        ):
        
        self.pos_init = pos_init
        self.shape = shape
        self.setup()
        
        self.update_func = update_func
        
        self.setup_colors(colors_hsv)
        
        self.frq_min = frequencies[0]
        self.frq_max = frequencies[-1]
        self.frequencies = (self.frq_max + self.frq_min) / 2 + (self.frq_max - self.frq_min) / 2 * np.cos(2*np.pi * np.linspace(0, 1, self.num_stars, dtype=np.float64))

    def setup(self):
        
        if self.shape['type'] == 'grid':
            self.num_stars = self.shape['x_points'] * self.shape['y_points']
            self.stars_pos = create_grid(
                center_point=self.pos_init,
                x_points=self.shape['x_points'],
                y_points=self.shape['y_points'],
                x_length=self.shape['x_length'],
                y_length=self.shape['y_length']
            )
        
        elif self.shape['type'] == 'circle':
            self.num_stars = self.shape['points']
            theta = np.linspace(0, 2 * np.pi, self.num_stars, endpoint=False) + np.pi / 2
            self.stars_pos = np.array([
                [self.pos_init[0] + self.shape['radius'] * np.cos(t), 
                 self.pos_init[1] + self.shape['radius'] * np.sin(t)]
                for t in theta], dtype=np.float64)
        
        self.stars_vel = np.zeros((self.num_stars, 2), dtype=np.float64)
    
    def setup_colors(self, colors_hsv):
        (hmin, hmax), s, v = colors_hsv[0], colors_hsv[1], colors_hsv[2]

        if self.shape['type'] == 'grid':
            col_direction = self.shape.get('col_direction', 'y')
            if col_direction == 'y':
                h_range = (hmax + hmin) / 2 + (hmax - hmin) / 2 * np.cos(
                    np.linspace(0, 2 * np.pi, self.shape['y_points'], dtype=np.float64) + np.pi
                )
                hsv_range = np.stack([h_range, np.full_like(h_range, s), np.full_like(h_range, v)], axis=1)
                self.stars_col = np.zeros((self.num_stars, 3), dtype=np.float64)
                for i in range(self.shape['y_points']):
                    start_idx = i * self.shape['x_points']
                    end_idx = start_idx + self.shape['x_points']
                    hsv = hsv_range[i]
                    self.stars_col[start_idx:end_idx] = colors.hsv_to_rgb(hsv)
            elif col_direction == 'x':
                h_range = (hmax + hmin) / 2 + (hmax - hmin) / 2 * np.cos(
                    np.linspace(0, 2 * np.pi, self.shape['x_points'], dtype=np.float64) + np.pi
                )
                hsv_range = np.stack([h_range, np.full_like(h_range, s), np.full_like(h_range, v)], axis=1)
                self.stars_col = np.zeros((self.num_stars, 3), dtype=np.float64)
                for i in range(self.shape['x_points']):
                    hsv = hsv_range[i]
                    self.stars_col[i::self.shape['x_points']] = colors.hsv_to_rgb(hsv)
        
        elif self.shape['type'] == 'circle':
            h_range = (hmax + hmin) / 2 + (hmax - hmin) / 2 * np.cos(
                np.linspace(0, 2 * np.pi, self.num_stars, dtype=np.float64) + np.pi
            )
            hsv_range = np.stack([h_range, np.full_like(h_range, s), np.full_like(h_range, v)], axis=1)
            self.stars_col = colors.hsv_to_rgb(hsv_range)
    
    def update(self, dt, with_sound=False, mixer=None, time_vid=None):
        
        for i in range(self.num_stars):
            collision = self.update_func(dt, self.stars_pos[i], self.stars_vel[i])
            
            if collision and with_sound:
                bounce_soundwave = create_bounce_soundwave(
                    duration=3.0,
                    attack_duration=0.010,
                    release_epsilon=1e-6,
                    frequency=self.frequencies[i])
                mixer.add_sound(bounce_soundwave, time_vid)

def create_grid(center_point, x_points, y_points, x_length, y_length):
    x_start = center_point[0] - x_length / 2
    y_start = center_point[1] - y_length / 2
    x = np.linspace(x_start, x_start + x_length, x_points)
    y = np.linspace(y_start, y_start + y_length, y_points)
    xx, yy = np.meshgrid(x, y, indexing='xy')
    grid = np.column_stack((xx.ravel(), yy[::-1].ravel()))
    return grid