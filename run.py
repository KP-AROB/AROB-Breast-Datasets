import argparse
import logging
import os
import numpy as np
from src.pipeline.breast import BreastImageProcessingPipeline
from collections import Counter
from src.processors.image import ImageDatasetProcessor
from src.utils.dataset import get_datasets
from src.augmentations.classwise import ClasswiseAugmentor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kaptios - Breast Dataset Preparation")
    parser.add_argument("--name", type=str, default='vindr',
                        choices=['vindr', 'cbis', 'inbreast'])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default='./data')
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--n_augment", type=int, default=0)
    parser.add_argument("--augment_type", type=str,
                        default='all',  choices=['all', 'geometric', 'photometric'])
    args = parser.parse_args()
    parser.set_defaults(synthetize=False)

    # INIT
    logging_message = "[KAPTIOS-2025-BREAST-DATASETS]"

    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - {logging_message} - %(levelname)s - %(message)s'
    )
    logging.info(f'Running {args.name} dataset preparation')

    image_pipeline = BreastImageProcessingPipeline()

    train_dataset, test_dataset = get_datasets(
        args.name, args.data_dir, args.task, image_pipeline)

    logging.info(f'Training dataset length : {len(train_dataset)}')
    logging.info(f'Test dataset length : {len(test_dataset)}')
    logging.info(f'Class balance : {Counter(train_dataset.targets)}')

    save_dir = os.path.join(args.out_dir, args.name, args.task)

    processor = ImageDatasetProcessor(
        train_dataset,
        test_dataset,
        save_dir,
        4
    )
    processor.run()

    if args.n_augment > 0:
        augmentor = ClasswiseAugmentor(save_dir + "/train", args.n_augment, np.unique(
            train_dataset.targets), args.augment_type)
        augmentor.run()

    logging.info('Preparation done.')
