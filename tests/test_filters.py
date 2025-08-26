import cv2
import numpy as np
from cvtools import filters

def test_convolution_identity():
    img = np.eye(5, dtype=np.uint8)*255
    kernel = np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float32)
    out = filters.convolution(img, kernel)
    assert np.allclose(out, img), "Convolución con kernel identidad debe dejar igual"

def test_sobel_shapes():
    img = np.random.randint(0,255,(50,50),dtype=np.uint8)
    sx = filters.sobel_x(img)
    sy = filters.sobel_y(img)
    assert sx.shape == img.shape
    assert sy.shape == img.shape

def test_canny_and_laplacian():
    img = np.zeros((50,50), dtype=np.uint8)
    cv2.rectangle(img,(10,10),(40,40),255,-1)
    edges = filters.canny_edge(img)
    lap = filters.laplacian(img)
    assert edges.shape == img.shape
    assert lap.shape == img.shape
    assert edges.max() > 0, "Debe detectar bordes"

