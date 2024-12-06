import numpy as np
import cv2


def normalize_int8(image: np.array):
    return cv2.normalize(
        image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    ).astype(np.uint8)


def truncate_normalization(image_mask: tuple):
    """Normalize an image within a given ROI mask

    Args:
        source (list): list of tuples containing cropped images and roi masks

    Returns:
        np.array: normalized image
    """
    img, mask = image_mask
    Pmin = np.percentile(img[mask != 0], 2)
    Pmax = np.percentile(img[mask != 0], 99)
    truncated = np.clip(img, Pmin, Pmax)
    normalized = (truncated - Pmin) / (Pmax - Pmin)
    normalized[mask == 0] = 0
    return normalized
