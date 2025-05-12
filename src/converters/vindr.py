from .base import BaseH5Converter
from src.loaders.vindr import VindrDataframeLoader
import os
import logging
import h5py
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        self.output_dir = os.path.join(self.output_dir, split)
        os.makedirs(self.output_dir, exist_ok=True)

        self.birads_dict = self.df['breast_birads'].to_dict()
        self.lesions_dict = self.df['finding_categories'].to_dict()

    def _process_dicom_image(self, path: str) -> Optional[Tuple[str, np.ndarray]]:
        try:
            image = resize_square(normalize_int8(read_dicom(path)), new_size=self.img_size)
            return image
        except Exception as e:
            logging.warning(f"Failed to process {path}: {e}")
            return None 

    def _process_dicom_image_in_parallel(self, image_paths: List[str]) -> Tuple[List[str], List[np.ndarray]]:
        images = []

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = {executor.submit(self._process_dicom_image, path): path for path in image_paths}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
                result = future.result()
                if result is not None:
                    images.append(result)
        return images

    def save_data_to_h5(self,  
        h5_filename: str, 
        image_batch: List[np.ndarray], 
        birads_batch: List[int], 
        lesions_batch: List[int]):
        with h5py.File(h5_filename, 'w') as h5_file:
            h5_file.create_dataset("images", data=np.array(image_batch),  compression="gzip")
            birads_dataset = h5_file.create_dataset("birads_labels", data=np.array(birads_batch, dtype=np.int32))
            lesions_dataset = h5_file.create_dataset("lesions_labels", data=np.array(lesions_batch, dtype=np.int32))
            birads_dataset.attrs['label_mapping'] = json.dumps(self.df_loader.birads_mapping)
            lesions_dataset.attrs['label_mapping'] = json.dumps(self.df_loader.lesions_mapping)

    def convert(self, split: str = 'training'):
        self._init_df(split)
        all_dicom_paths = self.df.index.tolist()
        total_files = len(all_dicom_paths)
        num_hdf5_files = ceil(total_files / self.chunk_size)

        logging.info(f"Total DICOM files found: {total_files}")
        logging.info(f"File chunk size: {self.chunk_size}")
        logging.info(f"Target HDF5 files to create: {num_hdf5_files}")

        chunks = [all_dicom_paths[i * self.chunk_size:(i + 1) * self.chunk_size] for i in range(num_hdf5_files)]

        for i, chunk in enumerate(chunks, start=1):
            logging.info(f"Processing chunk {i}/{num_hdf5_files}...")
            images = self._process_dicom_image_in_parallel(chunk)
            birads = [self.birads_dict[path] for path in chunk]
            lesions = [self.lesions_dict[path] for path in chunk]
                
            if not images:
                logging.warning(f"Chunk {i} is empty after filtering. Skipping.")
                continue
            hdf5_filename = os.path.join(self.output_dir, f"chunk_{i:04d}.h5")
            self.save_data_to_h5(hdf5_filename, images, birads, lesions)
            logging.info(f"Saved chunk {i} to {hdf5_filename}")
