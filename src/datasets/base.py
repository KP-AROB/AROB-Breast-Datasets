from torch.utils.data import Dataset
from src.pipeline.base import BasePipeline


class AbstractBreastDataset(Dataset):

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
