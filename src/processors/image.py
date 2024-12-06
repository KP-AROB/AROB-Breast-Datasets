import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import List

class ImageDatasetProcessor(object):

    def __init__(self, train_dataset: Dataset, test_dataset: Dataset, save_dir: str, batch_size = 32):
        
        shutil.rmtree(save_dir, ignore_errors=True)
        self.save_dir = save_dir
        self.train_loader = DataLoader(train_dataset, batch_size, shuffle = False, num_workers=4, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size, shuffle = False, num_workers=4, pin_memory=True)
        self.__create_out_directories(np.unique(train_dataset.targets))

    def __create_out_directories(self, classes: List[str]):
        for cls in classes:
            os.makedirs(os.path.join(self.save_dir, 'train', cls))
            os.makedirs(os.path.join(self.save_dir, 'test', cls))

    def run(self):
        with tqdm(total=len(self.train_loader), desc='Preparing train dataset') as pbar:
            for batch_idx, batch in enumerate(self.train_loader):
                images, labels = batch
                for img_idx, (img, label) in enumerate(zip(images, labels)):
                    out_file_path = os.path.join(self.save_dir, 'train', label, f'{batch_idx}_{img_idx}.png')
                    cv2.imwrite(out_file_path, img.cpu().numpy())
                pbar.update(1)

        with tqdm(total=len(self.train_loader), desc='Preparing test dataset') as pbar:
             for batch_idx, batch in enumerate(self.test_loader):
                images, labels = batch
                for img_idx, (img, label) in enumerate(zip(images, labels)):
                    out_file_path = os.path.join(self.save_dir, 'test', label, f'{batch_idx}_{img_idx}.png')
                    cv2.imwrite(out_file_path, img.cpu().numpy())
                pbar.update(1)
