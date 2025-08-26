import numpy as np

def radial_distortion(points, k1=0.0, k2=0.0):
    """
    Aplica distorsión radial a puntos proyectados.
    points: array Nx2 con coordenadas [x, y] normalizadas.
    """
    x, y = points[:, 0], points[:, 1]
    r2 = x**2 + y**2
    factor = 1 + k1*r2 + k2*(r2**2)
    x_dist = x * factor
    y_dist = y * factor
    return np.vstack([x_dist, y_dist]).T

def change_focal_length(points_3d, f):
    """
    Proyecta puntos 3D al plano de imagen variando longitud focal.
    points_3d: array Nx3 con puntos [X,Y,Z]
    f: longitud focal
    """
    X, Y, Z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    x_proj = f * (X / Z)
    y_proj = f * (Y / Z)
    return np.vstack([x_proj, y_proj]).T
