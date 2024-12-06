from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from src.pipeline.base import BasePipeline

class ImageDatasetProcessor(object):

    def __init__(self, train_dataset: Dataset, test_dataset: Dataset, batch_size = 32):
        self.train_loader = DataLoader(train_dataset, batch_size, shuffle = False, pin_memory=True, num_workers=4)
        self.test_loader = DataLoader(test_dataset, batch_size, shuffle = False, pin_memory=True, num_workers=4)

    def run(self):
        with tqdm(total=len(self.train_loader), desc='Preparing training dataset') as pbar:
            for x, y in self.train_loader:
                pbar.update(1)
            
