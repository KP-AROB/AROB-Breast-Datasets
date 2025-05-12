from .base import BaseH5Converter
from src.loaders.vindr import VindrDataframeLoader
import os
import logging
import h5py
import numpy as np
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Optional, List
from src.operations.read import read_dicom
from src.operations.normalize import normalize_int8
from src.operations.transform import resize_square
from math import ceil
from tqdm import tqdm

class VindrH5Converter(BaseH5Converter):
    def __init__(self, data_dir: str, output_dir: str, img_size: int = 224, chunk_size: int = 1000, num_threads: int = 4):
        super().__init__(data_dir, output_dir, chunk_size, num_threads) 
        self.img_size = img_size
        self.df_loader = VindrDataframeLoader(data_dir)

    def _init_df(self, split: str):
        self.df = self.df_loader(split=split)
        self.df.set_index('absolute_path', inplace=True)
    
        self.birads_dict = self.df['breast_birads'].to_dict()
        self.lesions_dict = self.df['finding_categories'].to_dict()

    def _process_dicom_image(self, path: str) -> Optional[Tuple[str, np.ndarray]]:
        try:
            image = resize_square(normalize_int8(read_dicom(path)), new_size=self.img_size)
            return image
        except Exception as e:
            logging.warning(f"Failed to process {path}: {e}")
            return None 

    async def _process_dicom_image_async(self, path, executor):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self._process_dicom_image, path)

    def save_data_to_h5(self,  
        index: int, 
        image_batch: List[np.ndarray], 
        birads_batch: List[int], 
        lesions_batch: List[int]):
        hdf5_filename = os.path.join(self.output_dir, f"chunk_{index:04d}.h5")
        with h5py.File(hdf5_filename, 'w') as h5_file:
            h5_file.create_dataset("images", data=np.array(image_batch),  compression="gzip")
            birads_dataset = h5_file.create_dataset("birads_labels", data=np.array(birads_batch, dtype=np.int32))
            lesions_dataset = h5_file.create_dataset("lesions_labels", data=np.array(lesions_batch, dtype=np.int32))
            birads_dataset.attrs['label_mapping'] = json.dumps(self.df_loader.birads_mapping)
            lesions_dataset.attrs['label_mapping'] = json.dumps(self.df_loader.lesions_mapping)
        logging.info(f"Saved chunk {index} to {hdf5_filename}")
    
    async def process_dicom_files(self, paths):
        executor = ThreadPoolExecutor(max_workers=self.num_threads) 
        batch_idx = 0
        
        image_batch = []
        birads_batch = [self.birads_dict[path] for path in paths]
        lesions_batch = [self.lesions_dict[path] for path in paths]
            
        for i, path in enumerate(paths):
            image = await self._process_dicom_image_async(path, executor)
            image_batch.append(image)
            
            if len(image_batch) >= self.chunk_size:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.save_data_to_h5, batch_idx, image_batch, birads_batch, lesions_batch)
                image_batch, birads_batch, lesions_batch = [], [], []
                batch_idx += 1

        if image_batch:
            await asyncio.get_running_loop().run_in_executor(None, self.save_data_to_h5, batch_idx, image_batch, birads_batch, lesions_batch)


    def convert(self, split: str = 'training'):
        self._init_df(split)
        self.output_dir = os.path.join(self.output_dir, split)
        os.makedirs(self.output_dir, exist_ok=True)
        all_dicom_paths = self.df.index.tolist()
        total_files = len(all_dicom_paths)
        num_hdf5_files = ceil(total_files / self.chunk_size)

        logging.info(f"Total DICOM files found: {total_files}")
        logging.info(f"File chunk size: {self.chunk_size}")
        logging.info(f"Target HDF5 files to create: {num_hdf5_files}")

        asyncio.run(self.process_dicom_files(all_dicom_paths))

