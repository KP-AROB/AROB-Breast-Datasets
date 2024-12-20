import pandas as pd
import os
import ast
import logging
from torch.utils.data import Dataset
from src.pipeline.base import BasePipeline


class VindrDataframeLoader(object):

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.birads_mapping = {
            'bi-rads_1': 'bi-rads_1',
            'bi-rads_2': 'bi-rads_2',
            'bi-rads_3': 'bi-rads_0',
            'bi-rads_4': 'bi-rads_0',
            'bi-rads_5': 'bi-rads_0'
        }

    def format_char(self, char):
        return char.lower().replace(' ', '_')

    def format_category_list(self, category_list):
        return [self.format_char(category) for category in category_list]

    def contains_all_classes(self, category_list, class_list):
        return any(cls in category_list for cls in class_list)

    def replace_categories(self, df, column, target_categories):
        def replace_if_present(categories):
            for target in target_categories:
                if target in categories:
                    return target
            return categories

        df[column] = df[column].apply(
            lambda x: replace_if_present(x) if isinstance(x, list) else x)

    def load_df(self, is_train=True):
        df_find = pd.read_csv(os.path.join(
            self.data_dir, 'finding_annotations.csv'))
        df_find['finding_categories'] = df_find['finding_categories'].apply(
            ast.literal_eval)
        df_find['finding_categories'] = df_find['finding_categories'].apply(
            self.format_category_list)
        df_find['breast_birads'] = df_find['breast_birads'].apply(
            self.format_char)
        df_find['breast_birads'] = df_find['breast_birads'].replace(
            self.birads_mapping)
        df_find.drop_duplicates(subset='image_id', keep=False, inplace=True)
        split_name = 'training' if is_train else 'test'
        df_find = df_find[df_find['split'] == split_name]
        return df_find


class VindrLesionDataset(Dataset):

    def __init__(self, data_dir: str, pipeline: BasePipeline, is_train: bool = True, task: str = 'lesions'):
        self.data_dir = data_dir
        self.is_train = is_train
        self.pipeline = pipeline
        self.task = task
        self.class_list = ['no_finding', 'mass', 'suspicious_calcification']
        self.df = self.load_dataframe()
        self.targets = self.df['finding_categories'].values if task == 'lesions' else self.df['breast_birads'].values

    def load_dataframe(self):
        df_loader = VindrDataframeLoader(self.data_dir)
        df = df_loader.load_df(self.is_train)
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
