import numpy as np
import glfw

from animengine.opengl import Window

def get_perspective_projection_matrix(fov_deg, aspect_ratio, near_plane, far_plane, dtype=np.float64):
    fov_rad = np.deg2rad(fov_deg)
    f = 1.0 / np.tan(fov_rad / 2.0)
    a = aspect_ratio
    n = near_plane
    f_p = far_plane

    return np.array([
        [f/a, 0.0, 0.0, 0.0],
        [0.0,   f, 0.0, 0.0],
        [0.0, 0.0, (f_p + n) / (n - f_p), (2 * f_p * n) / (n - f_p)],
        [0.0, 0.0, -1.0, 0.0]
    ], dtype=dtype)

class Camera:
    
    def __init__(self, window: Window):
        
        self.window = window
        self.width = window.width
        self.height = window.height
        self.aspect_ratio = self.width / self.height
        
        self.pos = np.array([3.0, 3.0, 3.0])
        
        self.front = np.array([-1.0, -1.0, -1.0])
        self.build_system(self.front)
        self.calculate_view_matrix()
        
        self.polar = np.arcsin(self.front[2]) * 180.0 / np.pi
        self.azimuth = np.arctan2(self.front[1], self.front[0]) * 180.0 / np.pi
        
        self.lastX = self.width // 2
        self.lastY = self.height // 2
        self.sensitivity = 0.1
        self.firstMouse = True
        
        self.fov = 45.0
        self.projection_matrix = get_perspective_projection_matrix(self.fov, self.aspect_ratio, 0.1, 100.0)
        
        glfw.set_cursor_pos_callback(window._window, self.look_around)
        glfw.set_scroll_callback(window._window, self.zoom)
    
    def move_forward(self, cameraSpeed):
        self.pos += cameraSpeed * self.front
    
    def move_backward(self, cameraSpeed):
        self.pos -= cameraSpeed * self.front
    
    def move_left(self, cameraSpeed):
        self.pos -= cameraSpeed * self.right
    
    def move_right(self, cameraSpeed):
        self.pos += cameraSpeed * self.right
    
    def handle_camera_movement(self, deltaTime):
        cameraSpeed = 2.5 * deltaTime
        if self.window.is_key_down(glfw.KEY_W):
            self.move_forward(cameraSpeed)
        if self.window.is_key_down(glfw.KEY_S):
            self.move_backward(cameraSpeed)
        if self.window.is_key_down(glfw.KEY_A):
            self.move_left(cameraSpeed)
        if self.window.is_key_down(glfw.KEY_D):
            self.move_right(cameraSpeed)
        
        self.calculate_view_matrix()
    
    def look_around(self, window: glfw._GLFWwindow, xpos: float, ypos: float) -> None:
        
        if self.firstMouse:
            self.lastX = xpos
            self.lastY = ypos
            self.firstMouse = False
        xoffset = xpos - self.lastX
        yoffset = self.lastY - ypos
        self.lastX = xpos
        self.lastY = ypos
        xoffset *= self.sensitivity
        yoffset *= self.sensitivity
        
        self.azimuth -= xoffset
        self.polar += yoffset
        if self.polar > 89:
            self.polar = 89
        if self.polar < -89:
            self.polar = -89
            
        azimuth_rad = np.deg2rad(self.azimuth)
        polar_rad = np.deg2rad(self.polar)
        
        front = np.array([
            np.cos(polar_rad) * np.cos(azimuth_rad),
            np.cos(polar_rad) * np.sin(azimuth_rad),
            np.sin(polar_rad)
        ])
        
        self.build_system(front)
        self.calculate_view_matrix()
    
    def build_system(self, front):
        z = np.array([0.0, 0.0, 1.0])
        
        self.front = front / np.linalg.norm(front)
        
        up = z - np.dot(z, self.front) * self.front
        self.up = up / np.linalg.norm(up)
        
        right = np.cross(self.front, up)
        self.right = right / np.linalg.norm(right)
        
        self.S = np.column_stack((self.right, self.up, -self.front))
    
    def calculate_view_matrix(self):
        
        self.view = np.array([
            [self.S[:, 0][0], self.S[:, 0][1], self.S[:, 0][2], 0.0],
            [self.S[:, 1][0], self.S[:, 1][1], self.S[:, 1][2], 0.0],
            [self.S[:, 2][0], self.S[:, 2][1], self.S[:, 2][2], 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]) @ np.array([
            [1.0, 0.0, 0.0, -self.pos[0]],
            [0.0, 1.0, 0.0, -self.pos[1]],
            [0.0, 0.0, 1.0, -self.pos[2]],
            [0.0, 0.0, 0.0, 1.0]
        ])
    
    def zoom(self, window: glfw._GLFWwindow, xoffset: float, yoffset: float) -> None:
        self.fov -= yoffset
        if self.fov < 15.0:
            self.fov = 15.0
        if self.fov > 45.0:
            self.fov = 45.0
        self.projection_matrix = get_perspective_projection_matrix(self.fov, self.aspect_ratio, 0.1, 100.0)
