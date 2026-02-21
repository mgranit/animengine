import numpy as np
from .linalg import reflect

def generate_random_rectangle_distribution(n, center, width, height):
    x = np.random.uniform(center[0] - width / 2, center[0] + width / 2, n)
    y = np.random.uniform(center[1] - height / 2, center[1] + height / 2, n)
    return np.vstack((x, y)).T

def generate_grid_distribution(n, center, width, height):
    
    # Calculate optimal grid dimensions (as square as possible)
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    # Calculate spacing between points
    dx = width / (cols - 1) if cols > 1 else 0
    dy = height / (rows - 1) if rows > 1 else 0
    
    # Calculate starting position (bottom-left of rectangle)
    x_start = center[0] - width / 2
    y_start = center[1] - height / 2
    
    # Generate points
    points = []
    count = 0
    
    for i in range(rows):
        for j in range(cols):
            if count >= n:
                break
            
            x = x_start + j * dx
            y = y_start + i * dy
            points.append([x, y])
            count += 1
        
        if count >= n:
            break
    
    return np.array(points)

def generate_circle_distribution(n, center, r):
    """
    Generate n random points uniformly distributed within a circle.
    
    Parameters:
    - n: number of points
    - center: (x, y) center of the circle
    - r: radius of the circle
    
    Returns:
    - numpy array of shape (n, 2) with point coordinates
    """
    # Generate random angles
    theta = np.random.uniform(0, 2 * np.pi, n)
    
    # Generate random radii using sqrt to ensure uniform distribution
    # (without sqrt, points cluster toward center)
    radius = np.sqrt(np.random.uniform(0, 1, n)) * r
    
    # Convert to cartesian coordinates
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    
    return np.vstack((x, y)).T

def generate_circle_grid_distribution(n, center, r):
    """
    Generate n points arranged in a grid pattern within a circle.
    Points are evenly spaced in concentric rings.
    
    Parameters:
    - n: number of points
    - center: (x, y) center of the circle
    - r: radius of the circle
    
    Returns:
    - numpy array of shape (n, 2) with point coordinates
    """
    if n == 0:
        return np.empty((0, 2))
    
    if n == 1:
        return np.array([center])
    
    points = []
    
    # Add center point
    points.append(center)
    remaining = n - 1
    
    # Determine number of rings
    ring = 1
    while remaining > 0:
        # Number of points in this ring (approximate by circumference)
        points_in_ring = min(remaining, max(6 * ring, 1))
        
        # Radius of this ring
        ring_radius = r * ring / max(int(np.sqrt(n / np.pi)) + 1, 1)
        
        if ring_radius > r:
            ring_radius = r
        
        # Distribute points evenly around the ring
        angles = np.linspace(0, 2 * np.pi, points_in_ring, endpoint=False)
        
        for angle in angles:
            x = center[0] + ring_radius * np.cos(angle)
            y = center[1] + ring_radius * np.sin(angle)
            points.append([x, y])
        
        remaining -= points_in_ring
        ring += 1
    
    return np.array(points[:n])

class Circle:
    
    def __init__(self, radius):
        
        self.radius = radius
    
    def find_intersection(self, pos0, vel0, acc, dt, max_iters=50, tol=1e-8):
        
        t_min = 0.0
        t_max = dt
        
        for _ in range(max_iters):
            t = (t_min + t_max) / 2.0
            
            pos_mid = pos0 + vel0 * t + 0.5 * acc * t**2
            r_mid = np.linalg.norm(pos_mid)
            
            if abs(r_mid - self.radius) < tol and r_mid < self.radius:
                break
            
            if r_mid < self.radius:
                t_min = t
            else:
                t_max = t
        
        pos = pos0 + vel0 * t + 0.5 * acc * t**2
        vel = vel0 + acc * t
        
        # ensure the position is on the circle
        pos = pos / np.linalg.norm(pos) * self.radius
        
        return t, pos, vel

    def check_collision(self, dt, pos0, vel0, posF, velF, acc):
        
        if np.linalg.norm(posF) >= self.radius:
            
            # find intersection point with the circle
            t_intersection, pos_intersection, vel_intersection = self.find_intersection(pos0, vel0, acc, dt)
            
            # reflect the velocity at the intersection point
            normal = pos_intersection / np.linalg.norm(pos_intersection)
            vel_intersection = reflect(-vel_intersection, normal)
            
            # calculate the remaining time and position after the intersection
            t_remain = dt - t_intersection
            
            # update position and velocity after the intersection
            posF = pos_intersection + vel_intersection * t_remain + 0.5 * acc * t_remain**2
            velF = vel_intersection + acc * t_remain
            
            return True, posF, velF
        
        else:
            return False, posF, velF

class PerforatedCircle(Circle):
    def __init__(self, radius, hole_num, hole_size):
        super().__init__(radius)
        
        self.holes = []
        for i in range(hole_num):
            start = i * 2 * np.pi / hole_num
            end = start + hole_size
            self.holes.append((start, end))

        self.holes_init = self.holes.copy()
    
    def rotate(self, angle):
        self.holes += np.array([angle, angle], dtype=np.float64)
        self.holes = np.mod(self.holes, 2 * np.pi)
    
    def angle_in_any_hole(self, theta):
        for start, end in self.holes:
            if start <= theta < end or (start > end and (theta >= start or theta < end)):
                return True
        return False
    
    def check_collision(self, dt, pos0, vel0, posF, velF, acc):
        
        theta = np.arctan2(posF[1], posF[0])
        if theta < 0: theta += 2 * np.pi
        hole_angle = self.angle_in_any_hole(theta)
        
        if np.linalg.norm(pos0) < self.radius and np.linalg.norm(posF) >= self.radius and not hole_angle:
            
            # find intersection point with the circle
            t_intersection, pos_intersection, vel_intersection = self.find_intersection(pos0, vel0, acc, dt)
            
            # reflect the velocity at the intersection point
            normal = pos_intersection / np.linalg.norm(pos_intersection)
            vel_intersection = reflect(-vel_intersection, normal)
            
            # calculate the remaining time and position after the intersection
            t_remain = dt - t_intersection
            
            # update position and velocity after the intersection
            posF = pos_intersection + vel_intersection * t_remain + 0.5 * acc * t_remain**2
            velF = vel_intersection + acc * t_remain
            
            return True, posF, velF
        
        else:
            return False, posF, velF
    
    def reset(self):
        self.holes = self.holes_init.copy()

class Parabola:
    def __init__(self, a, b, c):
        
        self.a = a
        self.b = b
        self.c = c
        
        self.params = np.array([a, b, c], dtype=np.float64)
    
    def f(self, x):
        return self.a * x**2 + self.b * x + self.c
    
    def df(self, x):
        return 2.0 * self.a * x + self.b
    
    def find_intersection(self, pos0, vel0, acc, dt, max_iters=50):
        
        t_min = 0.0
        t_max = dt
        
        for _ in range(max_iters):
            t = (t_min + t_max) / 2.0
            
            pos_mid = pos0 + vel0 * t + 0.5 * acc * t**2
            
            if abs(pos_mid[1] - self.f(pos_mid[0])) < 1e-8:
                break
            
            if pos_mid[1] > self.f(pos_mid[0]):
                t_min = t
            else:
                t_max = t
        
        pos = pos0 + vel0 * t + 0.5 * acc * t**2
        vel = vel0 + acc * t
        
        return t, pos, vel
    
    def resolve_collision(self, dt, pos0, vel0, posF, velF, acc):
        
        if posF[1] < self.f(posF[0]):
            
            # find intersection point with the parabola
            t_intersection, pos_intersection, vel_intersection = self.find_intersection(pos0, vel0, acc, dt)
            
            # find tangent vector at intersection point
            tangent = np.array([1.0, self.df(pos_intersection[0])], dtype=np.float64)
            
            # find normal vector at intersection point
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
            
            # reflect the velocity at the intersection point
            vel_intersection = reflect(-vel_intersection, normal)
            
            # calculate the remaining time and position after the intersection
            t_remain = dt - t_intersection

            # update position and velocity after the intersection
            posF = pos_intersection + vel_intersection * t_remain + 0.5 * acc * t_remain**2
            velF = vel_intersection + acc * t_remain
            
            return True, posF, velF
        
        else:
            
            return False, posF, velF