
from .base import BasePipeline
from src.operations.read import read_dicom
from src.operations.transform import crop_to_roi, resize_square
from src.operations.normalize import truncate_normalization, normalize_int8


class BreastImageProcessingPipeline(BasePipeline):
    def __init__(self):
        super().__init__()
        self.operations = [
            read_dicom,
            crop_to_roi,
            truncate_normalization,
            resize_square,
            normalize_int8
        ]
