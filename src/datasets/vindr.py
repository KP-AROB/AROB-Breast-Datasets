import os
import logging
from torch.utils.data import Dataset
from src.pipeline.base import BasePipeline
from src.loaders.vindr import VindrDataframeLoader

class VindrLesionDataset(Dataset):

    def __init__(self, data_dir: str, pipeline: BasePipeline, split: str = 'training', task: str = 'lesions'):
        self.data_dir = data_dir
        self.split = split
        self.pipeline = pipeline
        self.task = task
        self.class_list = ['no_finding', 'mass', 'suspicious_calcification']
        self.df = self.load_dataframe()
        self.targets = self.df['finding_categories'].values if task == 'lesions' else self.df['breast_birads'].values

    def load_dataframe(self):
        df_loader = VindrDataframeLoader(self.data_dir)
        df = df_loader(self.split)
        if self.task == 'lesions':
            df = df[df['finding_categories'].apply(
                lambda x: df_loader.contains_all_classes(x, self.class_list))]
            df_loader.replace_categories(
                df, 'finding_categories', self.class_list)
        else:
            mapping = {
                'bi-rads_1': 'bi-rads_1',
                'bi-rads_2': 'bi-rads_2',
                'bi-rads_3': 'bi-rads_0',
                'bi-rads_4': 'bi-rads_0',
                'bi-rads_5': 'bi-rads_0'
            }
            df['breast_birads'] = df['breast_birads'].replace(mapping)
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_path = os.path.join(
            self.data_dir, 'images', row['study_id'], row['image_id'] + '.dicom')
        try:
            image = self.pipeline.process(sample_path)
            label = self.targets[idx]
            return image, label

        except Exception as e:
            logging.info(
                'Could not process image {} - {}'.format(sample_path, e))
            return None

