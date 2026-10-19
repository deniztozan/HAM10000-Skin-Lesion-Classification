
import numpy as np
from skimage.feature import hog


def extract_hog(image, visualize=False):
    """
    Gri tonlamalı görüntüden HOG öznitelikleri çıkarır.
    """

    if image.ndim != 2:
        raise ValueError("HOG için gri tonlamalı görüntü bekleniyor.")

    result = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=visualize,
        feature_vector=True
    )

    return result
