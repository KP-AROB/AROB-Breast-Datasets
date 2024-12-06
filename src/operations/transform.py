import cv2
import numpy as np


def crop_to_roi(img: np.array):
    """Crop mammogram to breast region.

    Args:
        img (np.array): The original image as a uint8 np.array

    Returns:
        tuple (np.array, np.array): (cropped_image, roi)
    """
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, breast_mask = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cnts, _ = cv2.findContours(
        breast_mask.astype(
            np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnt = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    yield img[y: y + h, x: x + w], breast_mask[y: y + h, x: x + w]


def resize(img: np.array):
    yield cv2.resize(
        img,
        (256, 256),
        interpolation=cv2.INTER_LINEAR,
    )
