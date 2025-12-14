import numpy as np


def five_point_stencil(arr):
    """Return (x-1, x+1, y-1, y+1) neighbor-shifted arrays. Periodic wrapping in y, pad-wrap in x"""
    xmin = np.vstack((arr[-2], arr[:-1]))  # shift right   (i-1 -> i)
    xplus = np.vstack((arr[1:], arr[-2]))  # shift left    (i+1 -> i)

    ymin = np.column_stack((arr[:, -1], arr[:, :-1]))  # shift up      (j-1 -> j)
    yplus = np.column_stack((arr[:, 1:], arr[:, 0]))  # shift down    (j+1 -> j)

    return xmin, xplus, ymin, yplus


def masked_neighbors_centered(p, fluid_mask):
    """Solid-safe neighbors for pressure Jacobi iteration in pressure poisson solve.
    Any solid neighbor gets replaced by the center value (homog. Neumann)."""
    p_im, p_ip, p_jm, p_jp = five_point_stencil(p)
    f_im, f_ip, f_jm, f_jp = five_point_stencil(fluid_mask)

    p_im = np.where(f_im, p_im, p)
    p_ip = np.where(f_ip, p_ip, p)
    p_jm = np.where(f_jm, p_jm, p)
    p_jp = np.where(f_jp, p_jp, p)
    return p_im, p_ip, p_jm, p_jp


def two_field_divergence(u, v, dx, dy):
    """∇·u on the given component fields using central differences"""
    div = np.empty_like(u)

    # bulk
    div[1:-1, 1:-1] = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dx) + (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dy)

    # edges
    div[1:-1, 0] = (u[2:, 0] - u[:-2, 0]) / (2 * dx) + (v[1:-1, 1] - v[1:-1, -1]) / (2 * dy) # periodic for bottom
    div[1:-1, -1] = (u[2:, -1] - u[:-2, -1]) / (2 * dx) + (v[1:-1, 0] - v[1:-1, -2]) / (2 * dy) # periodic for top

    # wrap pad on left and right for next operation
    div[0] = div[-2]
    div[-1] = div[1]

    return div


def speed_ramp(target_speed, total_steps, step):
    """Smoothly ramp inlet speeds"""
    return target_speed * 0.5 * (1 - np.cos(np.pi * step / total_steps))


def parameter_scaling(arr):
    """Map continuous control to (0,1) via sigmoid: 1/(1+exp(-4x))"""
    return 1 / (1 + np.exp(-4 * arr))


def load_state(state_fname):
    """Reconstructs all quantity fields and from a state saved using System.save_state()"""
    data = np.load(state_fname)
    return data['u'], data['v'], data['p'], data['X'], data['Y'], data['wing_points']


def generate_NACA4(code, n_points=50):
    """Takes a 4 digit string representing a NACA4 airfoil code. Returns
    n_points upper and lower surface control points for a NACA 4-digit airfoil. Control points are used
    to define a Wing() object if a NACA code is specified in definition."""

    if len(code) != 4 or not code.isdigit():
        raise ValueError("NACA4 code must be a string of four digits")

    m = int(code[0]) / 100 # Max camber
    p = int(code[1]) / 10 # Location of max camber
    t = int(code[2:]) / 100 # Max thickness


    x = np.linspace(0.0, 1.0, n_points)

    # Thickness distribution (standard 4-digit)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

    # Camber line
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)

    for i, xi in enumerate(x):
        if p == 0:  # symmetric airfoil (e.g., 0012)
            yc[i] = 0.0
            dyc_dx[i] = 0.0
            continue

        if xi < p:
            yc[i] = m / p**2 * (2 * p * xi - xi**2)
            dyc_dx[i] = 2 * m / p**2 * (p - xi)
        else:
            yc[i] = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * xi - xi**2)
            dyc_dx[i] = 2 * m / (1 - p) ** 2 * (p - xi)

    # Surface angle
    theta = np.arctan(dyc_dx)

    upper_x = x - yt * np.sin(theta)
    upper_heights = yc + yt * np.cos(theta)

    lower_x = x + yt * np.sin(theta)
    lower_heights = -(yc - yt * np.cos(theta))

    return upper_x, upper_heights, lower_x, lower_heights


def better_time(s):
    """Prints long times in seconds in a much more readable format"""
    s = int(s)
    h = 0
    m = 0

    if s >= 3600:
        h = s // 3600
        s = s % 3600

    if s >= 60:
        m = s // 60
        s = s % 60

    return f"{h} hrs, {m} mins, {s} secs"
