"""HAM10000 HOG feature extraction utilities.

Shared by Random Forest, Logistic Regression, OvR and XGBoost notebooks.
The input must be a 2-D grayscale image. Image loading, resizing and
segmentation masking belong to src.preprocessing, not this module.

Compatibility: ``extract_hog(image, visualize=False)`` returns a feature
vector; with ``visualize=True`` it returns ``(features, hog_image)``.
"""

from __future__ import annotations

import numpy as np
from skimage.feature import hog

# Preserve the settings used for the existing 128x128 and 300x225 HOG files.
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"


def extract_hog(image: np.ndarray, visualize: bool = False):
    """Extract HOG from one grayscale image.

    Parameters
    ----------
    image : ndarray
        Two-dimensional grayscale image, typically uint8 (0..255).
        For reproducibility with existing experiments, resize to (300, 225)
        beforehand using ``preprocess_image`` or ``preprocess_with_mask``.
    visualize : bool
        If True, also return a HOG visualization image.

    Returns
    -------
    ndarray or tuple[ndarray, ndarray]
        1-D HOG descriptor, or (descriptor, visualization).
        A grayscale image with shape (225, 300) produces 34,992 features.
    """
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(
            f"HOG requires a 2-D grayscale image; got shape {image.shape}. "
            "Convert BGR/RGB to grayscale during preprocessing."
        )
    if image.shape[0] < 16 or image.shape[1] < 16:
        raise ValueError(f"Image too small for 2x2 HOG blocks: {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError("Image contains NaN or infinity")

    return hog(
        image,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        visualize=visualize,
        feature_vector=True,
    )
