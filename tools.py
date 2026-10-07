import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage.transform import rotate
from skimage.filters import threshold_otsu

from tools import *

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


def dilate_image(image, kernel_size=4, iterations=1):

    kernel = np.ones(
        (kernel_size, kernel_size),
        np.uint8
    )

    dilated_image = cv2.dilate(
        image,
        kernel,
        iterations=iterations
    )

    return dilated_image

def crop_object(image):

    dilated_image = dilate_image(image)

    label_image = label(dilated_image > 0)
    regions = regionprops(label_image)

    if not regions:
        raise ValueError("No object found in image.")

    region = max(regions, key=lambda r: r.area)

    minr, minc, maxr, maxc = region.bbox

    return image[minr:maxr, minc:maxc]


def make_square_crop(image, margin=10):

    height, width = image.shape[:2]

    size = max(height, width) + 2 * margin

    square = np.zeros(
        (size, size),
        dtype=image.dtype
    )

    y = (size - height) // 2
    x = (size - width) // 2

    square[
        y:y + height,
        x:x + width
    ] = image

    return square

def rotate_image(image, angle):

    return rotate(
        image,
        angle,
        resize=True,
        preserve_range=True
    )

def prepare_image(image, margin=10):

    cropped = crop_object(image)

    square = make_square_crop(
        cropped,
        margin=margin
    )

    return square



def calculate_fractal_dimension(image, n_rotations=20, seed=None):
    """
    Calculate the mean fractal dimension of an image
    over multiple random rotations.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale image.

    n_rotations : int, optional
        Number of random rotations used to calculate
        the mean fractal dimension.

    seed : int or None, optional
        Seed for reproducibility of the random angles.

    Returns
    -------
    float
        Mean fractal dimension.
    """

    # Prepare image
    image = np.invert(image)

    threshold = threshold_otsu(image)
    binary_image = ((image > threshold) * 255).astype(np.uint8)

    binary_image = make_square(binary_image)
    binary_image = add_one_pixel_border(binary_image)

    prepared_image = prepare_image(binary_image)

    # Generate random angles
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 360, n_rotations)

    # Calculate fractal dimension
    fractal_dimensions = []

    for angle in angles:
        rotated_image = rotate_image(
            prepared_image,
            angle
        )

        rotated_image = prepare_image(rotated_image)

        D = fractal_dimension(rotated_image)

        fractal_dimensions.append(D)

    return np.mean(fractal_dimensions)