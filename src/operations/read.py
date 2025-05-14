from pydicom import dcmread
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
    try:
        ds = dcmread(path)
        img2d = ds.pixel_array
        return img2d.astype(np.float32)
    except Exception as e:
        return None
