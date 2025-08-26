import cv2
import numpy as np

def convolution(img, kernel):
    """Convolución genérica."""
    return cv2.filter2D(img, -1, kernel)

def sobel_x(img):
    """Sobel en X."""
    return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

def sobel_y(img):
    """Sobel en Y."""
    return cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

def canny_edge(img, t1=100, t2=200):
    """Detector de bordes Canny."""
    return cv2.Canny(img, t1, t2)

def laplacian(img):
    """Filtro Laplaciano: resalta bordes y regiones con cambios bruscos."""
    return cv2.Laplacian(img, cv2.CV_64F)

