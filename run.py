import argparse
import logging
from src.datasets.vindr import VindrLesionDataset
from src.pipeline.breast import BreastImageProcessingPipeline
from collections import Counter
from src.processors.image import ImageDatasetProcessor
from src.datasets.cbis import CBISMetadataCorrector

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kaptios - Breast Dataset Preparation")
    parser.add_argument("--name", type=str, default='vindr', choices=['vindr', 'cbis', 'inbreast'])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default='./data')
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

    if args.name == "vindr":
        vindr_tasks = ['lesions', 'birads', 'anomalies']
        if args.task not in vindr_tasks:
            raise ValueError('task must be of {}'.format(vindr_tasks))
        train_dataset = VindrLesionDataset(args.data_dir, image_pipeline, True)
        test_dataset = VindrLesionDataset(args.data_dir, image_pipeline, False)
    elif args.name == "cbis":
        cbis_tasks = [
            'scan', 
            'scan-severity', 
            'scan-mass-severity',
            'scan-calc-severity',
            'roi-severity',
            'roi-mass-severity',
            'roi-calc-severity'
        ]
        if args.task not in cbis_tasks:
            raise ValueError('task must be of {}'.format(cbis_tasks))
    elif args.name == "inbreast":
        pass
    

    logging.info(f'Training dataset length : {len(train_dataset)}')
    logging.info(f'Test dataset length : {len(test_dataset)}')
    logging.info(f'Class balance : {Counter(train_dataset.targets)}')

    processor = ImageDatasetProcessor(
        train_dataset,
        test_dataset,
        args.out_dir,
        32
    )

    processor.run()