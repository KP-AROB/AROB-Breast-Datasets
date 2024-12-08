from src.datasets import *
from src.pipeline.base import BasePipeline
from torch.utils.data import random_split


def get_datasets(name: str, data_dir: str, task: str, pipeline: BasePipeline):
    if name == "vindr":
        vindr_tasks = ['lesions']
        if task not in vindr_tasks:
            raise ValueError('task must be of {}'.format(vindr_tasks))
        train_dataset = VindrLesionDataset(data_dir, pipeline, True)
        test_dataset = VindrLesionDataset(data_dir, pipeline, False)
    elif name == "cbis":
        cbis_tasks = [
            'lesions',
            'lesions-severity'
        ]

        if task not in cbis_tasks:
            raise ValueError('task must be of {}'.format(cbis_tasks))

        train_dataset = CBISDataset(data_dir, pipeline, task, True)
        test_dataset = CBISDataset(data_dir, pipeline, task, False)

    elif name == "inbreast":
        inbreast_tasks = ['lesions', 'birads']
        if task not in inbreast_tasks:
            raise ValueError('task must be of {}'.format(inbreast_tasks))
        dataset = InbreastDataset(data_dir, pipeline, task)
        original_targets = dataset.targets
        total_dataset_size = len(dataset)
        test_size = int(total_dataset_size * 0.2)
        train_size = total_dataset_size - test_size
        train_dataset, test_dataset = random_split(
            dataset, [train_size, test_size])
        train_targets = [original_targets[idx]
                         for idx in train_dataset.indices]
        test_targets = [original_targets[idx] for idx in test_dataset.indices]
        train_dataset.targets = train_targets
        test_dataset.targets = test_targets

    return train_dataset, test_dataset
