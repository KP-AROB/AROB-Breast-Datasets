from src.datasets.vindr import VindrLesionDataset
from src.pipeline.breast import BreastImageProcessingPipeline

pipeline = BreastImageProcessingPipeline()
dataset = VindrLesionDataset(
    data_dir='/mnt/d/datasets/vindr-mammo/data/',
    pipeline=pipeline
)
