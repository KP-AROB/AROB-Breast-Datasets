import logging, os, argparse
from src.processors.h5 import VindrH5BatchProcessor

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
    description="Kaptios - Breast Dataset Preparation")
    parser.add_argument("--name", type=str, default='vindr',
                        choices=['vindr', 'cbis', 'inbreast'])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--tmp_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=100)
    args = parser.parse_args()

    logging_message = "[KP-2025-BREAST-DATASETS]"
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - {logging_message} - %(levelname)s - %(message)s'
    )

    splits = ['test', 'training']
    
    processor = VindrH5BatchProcessor(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        n_workers=2 * os.cpu_count(),
        tmp_dir=args.tmp_dir or None
    )
    
    for i in splits:
        logging.info(f'Processing {i} split')
        processor.run(os.path.join(args.out_dir, i), i)

    logging.info('\nProcessing done.')