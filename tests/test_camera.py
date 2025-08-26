import numpy as np
from cvtools import camera

def test_radial_distortion_identity():
    pts = np.array([[0.5, 0.5], [0.0, 1.0]])
    distorted = camera.radial_distortion(pts, k1=0, k2=0)
    assert np.allclose(distorted, pts), "Sin distorsión debe dar igual"

def test_radial_distortion_expansion():
    pts = np.array([[1.0, 0.0]])
    distorted = camera.radial_distortion(pts, k1=0.1, k2=0.0)
    assert distorted[0,0] > 1.0, "El punto en x debe expandirse con k1 positivo"

def test_change_focal_length():
    pts_3d = np.array([[1, 2, 5]])
    proj_small_f = camera.change_focal_length(pts_3d, f=100)
    proj_large_f = camera.change_focal_length(pts_3d, f=1000)
    assert np.linalg.norm(proj_large_f) > np.linalg.norm(proj_small_f), \
        "Mayor focal debe proyectar más lejos del origen"

