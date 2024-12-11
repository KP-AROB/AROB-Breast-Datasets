import os
import cv2
import logging
import albumentations as A
from glob import glob
from tqdm import tqdm
from typing import List


def augment_image(image_path, n_augment, pipeline):
    image = cv2.imread(image_path)
    augmented_images = []
    for _ in range(n_augment):
        augmented = pipeline(image=image)['image']
        augmented_images.append(augmented)
    return augmented_images


class ClasswiseAugmentor(object):
    """Augmentor class to create new instances of images on under-represented classes"""

    def __init__(self, data_dir: str, n_augment: int, class_list: List[str], augmentation_type: str):
        """Classwise Augmentor constructor

        Args:
            data_dir (str): Directory of the dataset to augment
            n_augment (int): Number of new images to create per instance
            class_list (List[str]): Classes to augment
            augmentation_type (str): Type of the augmentations (geometric, photometric)
        """
        self.data_dir = data_dir
        self.n_augment = n_augment
        self.class_list = class_list
        self.augmentation_type = augmentation_type

        geometric_pipeline = [
            A.Flip(p=1),
            A.ElasticTransform(p=0.3),
            A.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        ]

        photometric_pipeline = [
            A.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.1, p=1),
            A.GaussianBlur(blur_limit=(1, 3), p=0.5),
        ]

        if self.augmentation_type == 'geometric':
            self.augmentation_pipeline = A.Compose(geometric_pipeline)
        elif self.augmentation_type == 'photometric':
            self.augmentation_pipeline = A.Compose(photometric_pipeline)
        else:
            self.augmentation_pipeline = A.Compose(
                geometric_pipeline + photometric_pipeline)

    def run(self):
        logging.info("Running data augmentation")
        class_dirs = [os.path.join(self.data_dir, str(i))
                      for i in self.class_list]
        for cls_path in class_dirs:
            number_of_images = glob(os.path.join(cls_path, '*.png'))
            with tqdm(total=len(number_of_images), desc=f"Augmenting {cls_path}") as pbar:
                for idx, img in enumerate(number_of_images):
                    augmented_images = augment_image(
                        img, self.n_augment, self.augmentation_pipeline)
                    for j, augmented_image in enumerate(augmented_images):
                        output_path = os.path.join(
                            cls_path, f"aug_{idx}_{j}.png")
                        cv2.imwrite(output_path, augmented_image)
                    pbar.update()

        logging.info("Augmentations finished.")
