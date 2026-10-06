import cv2
import numpy as np

def add_one_pixel_border(image):
    border_image = cv2.copyMakeBorder(
        image,
        1,
        1,
        1,
        1,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    return border_image
    

def make_square(image):
    height, width = image.shape[:2]
    size = max(height, width)

    top = (size - height) // 2
    bottom = size - height - top
    left = (size - width) // 2
    right = size - width - left

    square_image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    return square_image



def fractal_dimension(image):
    

    height, width = image.shape[:2]
    

    # Box sizes: exclude 1 pixel and the full image size
    box_sizes = 2 ** np.arange(1, int(np.log2(height)))
    counts = []

    for box_size in box_sizes:

        count = 0

        for y in range(0, height, box_size):
            for x in range(0, width, box_size):

                box = image[
                    y:min(y + box_size, height),
                    x:min(x + box_size, width)
                ]

                if np.any(box):
                    count += 1

        counts.append(count)

    box_sizes = np.asarray(box_sizes)
    counts = np.asarray(counts)

    # Remove scales with zero occupied boxes
    valid = counts > 0

    box_sizes = box_sizes[valid]
    counts = counts[valid]

    # Linear regression in log-log space
    slope, _ = np.polyfit(
        np.log(1 / box_sizes),
        np.log(counts),
        1
    )

    return slope


def rotate_image(image, angle):
    height, width = image.shape[:2]

    center = (width / 2, height / 2)

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return rotated_image