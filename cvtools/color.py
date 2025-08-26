import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

def rgb_to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

def rgb_to_lab(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

def plot_histogram(img):
    """Grafica histograma de colores RGB."""
    colors = ('r', 'g', 'b')
    for i, col in enumerate(colors):
        plt.hist(img[:, :, i].ravel(), bins=256, color=col, alpha=0.6)
    plt.title("Histograma de colores")
    plt.show()

def quantize_image(img, num_colors=16):
    """Cuantización con KMeans (aprox Median Cut)."""
    h, w, c = img.shape
    pixels = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    new_colors = kmeans.cluster_centers_.astype(np.uint8)
    return new_colors[labels].reshape((h, w, 3))

def reduce_image_size(img, num_colors=16, out_path="temp.jpg"):
    """Reduce peso disminuyendo cantidad de colores y devuelve tamaño en KB."""
    quant_img = quantize_image(img, num_colors=num_colors)
    cv2.imwrite(out_path, cv2.cvtColor(quant_img, cv2.COLOR_RGB2BGR))
    size_kb = os.path.getsize(out_path) / 1024
    return quant_img, size_kb

