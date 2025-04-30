import logging, os
from src.converters.vindr import VindrH5Converter

logging_message = "[KAPTIOS-2025-BREAST-DATASETS]"

logging.basicConfig(
    level=logging.INFO,
    format=f'%(asctime)s - {logging_message} - %(levelname)s - %(message)s'
)

available_cpus = os.cpu_count()
logging.info(f"Available CPUs: {available_cpus}")
data_path = '/media/nvidia/DATA1/Data/original_breast_datasets/vindr_mammo/'
output_path = './data/vindr_mammo_h5/'

vindrConverter = VindrH5Converter(
    data_dir=data_path,
    output_dir=output_path,
    img_size=224,
    chunk_size=100,
    num_processes=available_cpus - 2
)

vindrConverter.convert('training')
