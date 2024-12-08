from torch.utils.data import Dataset
from src.pipeline.base import BasePipeline
from tqdm import tqdm
from glob import glob
import os
import pandas as pd
import logging


class CBISDataframeLoader(object):

    def __init__(self, data_dir: str, is_train: bool):
        self.data_dir = data_dir
        self.mode = 'train' if is_train else 'test'

        if len(glob(data_dir + '/*corrected.csv')) != 4:
            logging.info('Corrected csv files not found. Creating ...')
            self.correct_metadata_files(data_dir)
            logging.info(f'Corrected csv files saved at {data_dir}')

        self.df_mass = pd.read_csv(os.path.join(
            data_dir, f'mass_case_description_{self.mode}_set_corrected.csv'))
        self.df_calc = pd.read_csv(os.path.join(
            data_dir, f'calc_case_description_{self.mode}_set_corrected.csv'))
        self.make_cls_column()

    def make_cls_column(self):
        """Modify the pathology columns to have four unique class values
        """
        self.df_mass.loc[self.df_mass['pathology'] ==
                         'BENIGN_WITHOUT_CALLBACK', 'pathology'] = 'BENIGN'
        self.df_calc.loc[self.df_calc['pathology'] ==
                         'BENIGN_WITHOUT_CALLBACK', 'pathology'] = 'BENIGN'

        self.df_mass['pathology'] = self.df_mass['abnormality type'] + \
            '_' + self.df_mass['pathology']
        self.df_calc['pathology'] = self.df_calc['abnormality type'] + \
            '_' + self.df_calc['pathology']

    def normalize_and_format_path(self, path: str) -> str:
        if path.startswith(".\\"):
            path = path[2:]
        path = path.replace("\\", "/")
        path_parts = path.split("/")
        if path_parts:
            last_part = path_parts[-1]
            number, *rest = last_part.split("-", 1)
            if number.isdigit():
                number = number.zfill(2)
            path_parts[-1] = f"{number}-{''.join(rest)}"
        return "/".join(path_parts)

    def get_image_path_ids(self, row, key):
        path = row[key]
        path_segment = path.split(os.sep)
        study_id = path_segment[1]
        series_uid = path_segment[2]
        return study_id, series_uid

    def correct_metadata_files(self):

        metadata_df = pd.read_csv(os.path.join(self.data_dir, 'metadata.csv'))

        lesion_description_files = {
            f"{desc}_case_description_{set_type}_set": os.path.join(self.data_dir, f"{desc}_case_description_{set_type}_set.csv")
            for desc in ["mass", "calc"]
            for set_type in ["train", "test"]
        }

        with tqdm(total=len(lesion_description_files.keys()), desc='Correcting csv files') as pbar:
            for key in lesion_description_files.keys():
                df = pd.read_csv(lesion_description_files[key])
                df = df.rename(columns={
                    'left or right breast': 'left_or_right_breast',
                    'image view': 'image_view',
                    'abnormality id': 'abnormality_id',
                    'mass shape': 'mass_shape',
                    'mass margins': 'mass_margins',
                    'image file path': 'image_file_path',
                    'cropped image file path': 'cropped_image_file_path',
                    'ROI mask file path': 'roi_mask_file_path'})

                for idx, row in df.iterrows():
                    image_study_id, image_series_uid = self.get_image_path_ids(
                        row, 'image_file_path')
                    roi_study_id, roi_series_uid = self.get_image_path_ids(
                        row, 'roi_mask_file_path')
                    cropped_study_id, cropped_series_uid = self.get_image_path_ids(
                        row, 'cropped_image_file_path')

                    meta_image = metadata_df[(metadata_df['Series UID'] == image_series_uid) & (
                        metadata_df['Study UID'] == image_study_id)]
                    meta_roi = metadata_df[(metadata_df['Series UID'] == roi_series_uid) & (
                        metadata_df['Study UID'] == roi_study_id)]
                    meta_cropped = metadata_df[(metadata_df['Series UID'] == cropped_series_uid) & (
                        metadata_df['Study UID'] == cropped_study_id)]

                    correct_img_path = meta_image['File Location'].values[0]
                    correct_roi_path = meta_roi['File Location'].values[0]
                    correct_cropped_path = meta_cropped['File Location'].values[0]

                    df.loc[idx, 'image_file_path'] = self.normalize_and_format_path(
                        correct_img_path)
                    df.loc[idx, 'roi_mask_file_path'] = self.normalize_and_format_path(
                        correct_roi_path)
                    df.loc[idx, 'cropped_image_file_path'] = self.normalize_and_format_path(
                        correct_cropped_path)

                df.to_csv(os.path.join(self.data_dir, key + '_corrected.csv'))
                pbar.update()

        logging.info(f'Corrected csv files saved at {self.data_dir}')


class CBISDataset(Dataset):

    def __init__(self, data_dir: str, pipeline: BasePipeline, task: str, is_train: bool = True):
        super().__init__()
        self.data_dir = data_dir
        self.is_train = is_train
        self.pipeline = pipeline
        self.task = task
        self.df = self.load_dataframe()
        self.targets = self.df['abnormality type'].values if task == 'lesion' else self.df['pathology'].values

    def load_dataframe(self):
        df_loader = CBISDataframeLoader(self.data_dir, self.is_train)
        df_mass = df_loader.df_mass
        df_calc = df_loader.df_calc
        return pd.concat([df_mass, df_calc])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(
            self.data_dir, row['image_file_path'])
        image = glob(image_path + '/*.dcm')[0]
        image = self.pipeline.process(image)
        label = self.targets[idx]
        return image, label
