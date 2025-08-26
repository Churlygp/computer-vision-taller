import cv2
import matplotlib.pyplot as plt
import numpy as np
from cvtools import camera, color, filters

# ================================
# 1. Ejemplo módulo camera
# ================================
points_3d = np.array([[1,1,5],[2,2,10],[3,1,7],[0.5,2,4]])
proj1 = camera.change_focal_length(points_3d, f=500)
proj2 = camera.change_focal_length(points_3d, f=1000)

print("Proyección con f=500:", proj1)
print("Proyección con f=1000:", proj2)

# ================================
# 2. Ejemplo módulo color
# ================================
img = cv2.imread("data/ejemplo1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Histograma
color.plot_histogram(img)

# Cuantización
img_quant = color.quantize_image(img, num_colors=8)

# Guardar y medir peso
quant_img, size_kb = color.reduce_image_size(img, num_colors=8, out_path="data/quant.jpg")
print("Tamaño de imagen cuantizada:", f"{size_kb:.2f} KB")

# ================================
# 3. Ejemplo módulo filters
# ================================
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

sobel_x = filters.sobel_x(gray)
sobel_y = filters.sobel_y(gray)
canny = filters.canny_edge(gray)
laplace = filters.laplacian(gray)

plt.figure(figsize=(12,8))
plt.subplot(2,3,1); plt.imshow(img); plt.title("Original"); plt.axis("off")
plt.subplot(2,3,2); plt.imshow(sobel_x, cmap="gray"); plt.title("Sobel X"); plt.axis("off")
plt.subplot(2,3,3); plt.imshow(sobel_y, cmap="gray"); plt.title("Sobel Y"); plt.axis("off")
plt.subplot(2,3,4); plt.imshow(canny, cmap="gray"); plt.title("Canny"); plt.axis("off")
plt.subplot(2,3,5); plt.imshow(laplace, cmap="gray"); plt.title("Laplaciano"); plt.axis("off")
plt.show()

