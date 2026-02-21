import numpy as np

def get_spread(pos, num, spread, type_='x'):
    if type_ == 'x':
        return np.array([[x, pos[1]] for x in np.linspace(pos[0] - spread / 2, pos[0] + spread / 2, num)])
    elif type_ == 'x-parabolic':
        x_values = parabolic_spacing_center_peak(num, k=spread / 2, p=pos[0])
        return np.array([[x, pos[1]] for x in x_values])
    elif type_ == 'y':
        return np.array([[pos[0], y] for y in np.linspace(pos[1] - spread / 2, pos[1] + spread / 2, num)])
    elif type_ == 'circle':
        theta = np.linspace(0, 2 * np.pi, num, endpoint=False) + np.pi / 2
        return np.array([[pos[0] + spread / 2 * np.cos(t), pos[1] + spread / 2 * np.sin(t)] for t in theta])
    elif type_ == 'normal':
        return np.random.multivariate_normal(pos, spread, num)
    else:
        raise ValueError("Invalid type. Use 'x', 'y', or 'circle'.")

def gaussian(x, a, b, c):
    return a * np.exp(-0.5 * ((x - b) / c)**2)

def parabolic_spacing_center_peak(num, a=1.0, k=1.0, p=0.0):
    """
    Returns an array of `num` values between p-k and p+k, spaced such that their density
    peaks in the middle (at p) and drops off at the edges.
    """
    from scipy.integrate import quad
    from scipy.optimize import bisect
    import numpy as np

    # Ensure density is always positive by adding a constant offset
    c = a * 0.25  # since (x-0.5)^2 max is 0.25 at edges
    def density(x):
        return -a * (x - 0.5)**2 + c

    def cdf(x):
        result, _ = quad(density, 0, x)
        return result

    total, _ = quad(density, 0, 1)
    def norm_cdf(x):
        return cdf(x) / total

    values = []
    for i in range(num):
        q = i / (num - 1) if num > 1 else 0.0
        x = bisect(lambda x_: norm_cdf(x_) - q, 0, 1)
        # Map from [0, 1] to [p-k, p+k]
        x_mapped = p - k + x * 2 * k
        values.append(x_mapped)
    return np.array(values)