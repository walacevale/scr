import cv2
import matplotlib.pyplot as plt
import numpy as np
from tools import *
from skimage.filters import threshold_otsu


img = np.invert(cv2.imread('/data/SierpinskiTriangle.png', cv2.IMREAD_GRAYSCALE))
img_border = add_one_pixel_border(img)

thresh = threshold_otsu(img_border)
img_binary = ((img_border > thresh)*255).astype(np.uint8)

img_binary = make_square(img_binary)


angles = np.arange(0, 360, 30)
fractal_dimensions = []

for angle in angles:
    rotated_image = rotate_image(img_binary, angle)

    D = fractal_dimension(rotated_image)

    fractal_dimensions.append(D)

mean_D = np.mean(fractal_dimensions)
print(mean_D)