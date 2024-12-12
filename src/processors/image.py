import os
import shutil
import cv2
import numpy as np
import torch
import uuid
import concurrent.futures
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import List


class ImageDatasetProcessor(object):

    def __init__(self, train_dataset: Dataset, test_dataset: Dataset, save_dir: str, batch_size=8):

        shutil.rmtree(save_dir, ignore_errors=True)
        self.save_dir = save_dir
        self.train_loader = DataLoader(
            train_dataset, batch_size, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(
            test_dataset, batch_size, shuffle=False, num_workers=4)

        self.__create_out_directories(np.unique(train_dataset.targets))

    def __create_out_directories(self, classes: List[str]):
        for cls in classes:
            os.makedirs(os.path.join(self.save_dir,
                        'train', str(cls)), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir,
                        'test', str(cls)), exist_ok=True)

    def save_images_batch(self, images, labels, dataset_type):
        out_file_path = os.path.join(self.save_dir, dataset_type)
        os.makedirs(out_file_path, exist_ok=True)

        for img, label in zip(images, labels):
            image_id = str(uuid.uuid4())
            label_str = str(label.item()) if torch.is_tensor(
                label) else str(label)
            out_file_path = os.path.join(
                self.save_dir, dataset_type, label_str, f'{image_id}.png')
            cv2.imwrite(out_file_path, img.cpu().numpy())

    def process_batch(self, dataset_type, dataloader):
        with tqdm(total=len(dataloader), desc=f'Preparing {dataset_type} dataset') as pbar:
            for batch in dataloader:
                if batch:
                    images, labels = batch
                    self.save_images_batch(images, labels, dataset_type)
                pbar.update(1)

    def run(self):
        tasks = []
        dataloaders = {
            'train': self.train_loader,
            'test': self.test_loader
        }
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for name, loader in dataloaders.items():
                tasks.append(
                    executor.submit(self.process_batch,
                                    name, loader)
                )
            for task in concurrent.futures.as_completed(tasks):
                task.result()
