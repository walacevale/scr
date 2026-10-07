import cv2
from tools import calculate_fractal_dimension


img = cv2.imread(
    'data_test/SierpinskiTriangle.png',
    cv2.IMREAD_GRAYSCALE
)

mean_D = calculate_fractal_dimension(img)

print('Mean D:', mean_D)