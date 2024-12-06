import numpy as np
import cv2


def normalize_int8(img: np.array):
    yield cv2.normalize(
        img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    ).astype(np.uint8)


def truncate_normalization(source: tuple):
    """Normalize an image within a given ROI mask

    Args:
        (img, mask) tuple: cropped image and roi mask

    Returns:
        np.array: normalized image
    """
    img, mask = source
    Pmin = np.percentile(img[mask != 0], 2)
    Pmax = np.percentile(img[mask != 0], 99)
    truncated = np.clip(img, Pmin, Pmax)
    normalized = (truncated - Pmin) / (Pmax - Pmin)
    normalized[mask == 0] = 0
    yield normalized
