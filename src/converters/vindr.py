from .base import BaseH5Converter
from src.loaders.vindr import VindrDataframeLoader
import os
import logging
import numpy as np
import h5py
import json
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm, trange
from typing import List, Tuple, Optional
from src.operations.read import read_dicom
from src.operations.normalize import normalize_int8
from src.operations.transform import resize_square
from math import ceil


def process_dicom_image(path: str) -> Optional[Tuple[str, np.ndarray]]:
    try:
        image = resize_square(normalize_int8(read_dicom(path)), new_size=256)
        return path, image
    except Exception as e:
        logging.warning(f"Failed to process {path}: {e}")
        return None


class VindrH5Converter(BaseH5Converter):
    def __init__(self, data_dir: str, output_dir: str, chunk_size: int = 1000, num_processes: int = 2, split='training'):
        super().__init__(data_dir, output_dir, chunk_size, num_processes)
        self.df_loader = VindrDataframeLoader(data_dir)
        self.df = self.df_loader(split=split)
        self.df.set_index('absolute_path', inplace=True)
        self.output_dir = os.path.join(output_dir, split)
        os.makedirs(self.output_dir, exist_ok=True)

        self.birads_dict = self.df['breast_birads'].to_dict()
        self.lesions_dict = self.df['finding_categories'].to_dict()

    def _read_chunk(self, executor: ProcessPoolExecutor, paths: List[str]) -> Tuple[List[str], List[np.ndarray]]:
        results = list(tqdm(executor.map(process_dicom_image, paths),
                            total=len(paths), desc="Reading DICOMs", leave=False))

        results = [r for r in results if r is not None]
        if not results:
            return [], []

        processed_paths, pixel_arrays = zip(*results)
        return list(processed_paths), list(pixel_arrays)
    
    def convert(self):
        all_dicom_paths = self.df.index.tolist()
        total_files = len(all_dicom_paths)
        num_hdf5_files = ceil(total_files / self.chunk_size)

        logging.info(f"Total DICOM files found: {total_files}")
        logging.info(f"Target HDF5 files to create: {num_hdf5_files}")

        start_index = 0

        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            for i in trange(num_hdf5_files, desc="Converting chunks", unit="chunk"):
                end_index = min(start_index + self.chunk_size, total_files)
                chunk_paths = all_dicom_paths[start_index:end_index]
                hdf5_filename = os.path.join(self.output_dir, f"chunk_{i:04d}.h5")

                processed_paths, pixel_arrays = self._read_chunk(executor, chunk_paths)

                if not processed_paths:
                    logging.info("No valid DICOM images found in this chunk.")
                    start_index = end_index
                    continue

                birads = [self.birads_dict[path] for path in processed_paths]
                lesions = [self.lesions_dict[path] for path in processed_paths]

                try:
                    with h5py.File(hdf5_filename, 'w') as h5f:
                        h5f.create_dataset("images", data=np.stack(pixel_arrays), compression="lzf")
                        h5f.create_dataset("paths", data=np.array(processed_paths, dtype='S'))
                        birads_dataset = h5f.create_dataset("birads_labels", data=np.array(birads, dtype=np.int32))
                        lesions_dataset = h5f.create_dataset("lesions_labels", data=np.array(lesions, dtype=np.int32))
                        birads_dataset.attrs['label_mapping'] = json.dumps(self.df_loader.birads_mapping)
                        lesions_dataset.attrs['label_mapping'] = json.dumps(self.df_loader.lesions_mapping)
                    logging.info(f"HDF5 file saved: {hdf5_filename}")
                except Exception as e:
                    logging.info(f"Error writing HDF5 file {hdf5_filename}: {e}")

                start_index = end_index
