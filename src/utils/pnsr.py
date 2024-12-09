import numpy as np


def calculate_psnr(original, processed):
    if original.shape != processed.shape:
        raise ValueError("Input images must have the same dimensions")
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel_value = 255.0
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    return psnr
