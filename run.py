import argparse
import logging

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
