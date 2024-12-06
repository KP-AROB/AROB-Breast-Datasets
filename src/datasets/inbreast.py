import os
import pandas as pd
from torch.utils.data import Dataset
from src.pipeline.base import BasePipeline
from glob import glob

class InbreastDataset(Dataset):

    def __init__(self, data_dir: str, pipeline: BasePipeline, task = 'lesion'):
        super().__init__()
        self.data_dir = data_dir
        self.pipeline = pipeline
        self.task = task
        self.df = self.load_dataframe()
        self.targets = self.df['Lesion annotation status'].values

    def load_dataframe(self):
        df = pd.read_excel(os.path.join(self.data_dir, "INbreast.xls"), skipfooter=2)
        df.columns = df.columns.str.strip().str.capitalize()
        if self.task == 'birads':
            df = df[df["Bi-rads"].notna()]
        df["Lesion annotation status"] = df["Lesion annotation status"].fillna(1)
        df.loc[df["Lesion annotation status"] != 1, "Lesion annotation status"] = 0
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_path = glob(
            os.path.join(self.data_dir, "AllDICOMs", str(row["File name"]) + "*.dcm")
        )[0]
        image = self.pipeline.process(sample_path)
        label = row['Bi-rads'] if self.task == 'birads' else row["Lesion annotation status"]
        return image, label
