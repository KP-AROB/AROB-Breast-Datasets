from pydicom import dcmread
from pydicom.pixels import apply_voi_lut
import numpy as np


def read_dicom(path: str):
    """Read a dicom file and return its np.array representation;
    The method applies a VOI lookup table and invert pixels intensities when the PhotometricInterpretation of
    a file is found to be 'MONOCHROME1'.

    Args:
        path (str): Path to dicom file

    Returns:
        np.array: The loaded image as np.array
    """
    ds = dcmread(path)
    img2d = ds.pixel_array
    img2d = apply_voi_lut(img2d, ds)
    if ds.PhotometricInterpretation == "MONOCHROME1":
        img2d = np.amax(img2d) - img2d
    return img2d
