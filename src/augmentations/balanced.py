import os
import cv2
import logging
import albumentations as A
import concurrent.futures

from glob import glob
from tqdm import tqdm
from typing import List
from collections import Counter


def augment_image(image_path, n_augment, pipeline):
    image = cv2.imread(image_path)
    augmented_images = []
    for _ in range(n_augment):
        augmented = pipeline(image=image)['image']
        augmented_images.append(augmented)
    return augmented_images


class BalancedAugmentor(object):
    """This Augmentor class can be used to over-sample imbalanced dataset by creating new instances of under-represented classes"""

    def __init__(self, data_dir: str, dataset_targets: List[int]):
        """Classwise Augmentor constructor

        Args:
            data_dir (str): Directory of the dataset to augment
            n_augment (int): Number of new images to create per instance
            class_list (List[str]): Classes to augment
            augmentation_type (str): Type of the augmentations (geometric, photometric)
        """
        self.data_dir = data_dir

        self.dataset_targets_count = Counter(dataset_targets)
        self.max_count = max(self.dataset_targets_count.values())

        geometric_pipeline = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        ]

        photometric_pipeline = [
            A.ElasticTransform(p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.GaussianBlur(blur_limit=(1, 3), p=0.5),
        ]

        self.augmentation_pipeline = A.Compose(
            geometric_pipeline + photometric_pipeline)

    def augment_class(self, cls_path: str, n_augment: int):
        """Augment images for a specific class directory.

        Args:
            cls_path (str): Path to the class directory.
            n_augment (int): Number of augmentations to perform per image.
        """
        number_of_images = glob(os.path.join(cls_path, '*.png'))
        progress_desc = f"Augmenting {cls_path} - Ratio x{n_augment}"

        with tqdm(total=len(number_of_images), desc=progress_desc) as pbar:
            for idx, img in enumerate(number_of_images):
                augmented_images = augment_image(
                    img, n_augment, self.augmentation_pipeline)
                for j, augmented_image in enumerate(augmented_images):
                    output_path = os.path.join(
                        cls_path, f"aug_{idx}_{j}.png")
                    cv2.imwrite(output_path, augmented_image)
                pbar.update()

    def run(self):
        logging.info("Running data augmentation")

        under_sampled_classes = [
            class_label for class_label, count in self.dataset_targets_count.items() if count < self.max_count]

        class_dirs = [os.path.join(self.data_dir, str(i))
                      for i in under_sampled_classes]

        tasks = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for idx, cls_path in enumerate(class_dirs):
                class_count = self.dataset_targets_count[under_sampled_classes[idx]]
                n_augment = int(self.max_count / class_count)
                tasks.append(
                    executor.submit(self.augment_class,
                                    cls_path, n_augment)
                )
            for task in concurrent.futures.as_completed(tasks):
                task.result()

        logging.info("Augmentations finished.")
