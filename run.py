import argparse
import logging
import os
from src.pipeline.breast import BreastImageProcessingPipeline
from collections import Counter
from src.processors.image import ImageDatasetProcessor
from src.utils.dataset import get_datasets
from src.augmentations.balanced import BalancedAugmentor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kaptios - Breast Dataset Preparation")
    parser.add_argument("--name", type=str, default='vindr',
                        choices=['vindr', 'cbis', 'inbreast'])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default='./data')
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--augment", action='store_true')
    args = parser.parse_args()
    parser.set_defaults(augment=True)

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
        8
    )
    processor.run()

    if args.augment:
        augmentor = BalancedAugmentor(
            save_dir + "/train", train_dataset.targets)
        augmentor.run()

    logging.info('Preparation done.')
