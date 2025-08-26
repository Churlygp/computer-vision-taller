import cv2
import numpy as np
import os
from cvtools import color

def test_rgb_to_hsv_and_lab():
    img = np.zeros((10,10,3), dtype=np.uint8)
    hsv = color.rgb_to_hsv(img)
    lab = color.rgb_to_lab(img)
    assert hsv.shape == img.shape
    assert lab.shape == img.shape

def test_quantize_image():
    img = np.random.randint(0,255,(20,20,3),dtype=np.uint8)
    q = color.quantize_image(img, num_colors=4)
    assert q.shape == img.shape
    # Solo 4 colores posibles
    assert len(np.unique(q.reshape(-1,3), axis=0)) <= 4

def test_reduce_image_size(tmp_path):
    img = np.random.randint(0,255,(20,20,3),dtype=np.uint8)
    out_file = tmp_path / "out.jpg"
    qimg, size = color.reduce_image_size(img, num_colors=4, out_path=str(out_file))
    assert os.path.exists(out_file)
    assert size > 0
    assert qimg.shape == img.shape

