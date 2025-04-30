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
from contextlib import contextmanager

class HDF5Store(object):
    def __init__(self, datapath, datasets, shapes, dtype, compression="gzip", chunk_len=1):
        self.datapath = datapath
        self.datasets = datasets
        self.shapes  = shapes
        self.i = {dataset: 0 for dataset in datasets}

        with h5py.File(self.datapath, mode='w') as h5f:
            for idx, i in enumerate(datasets):
                if shapes[idx] == 0:
                    shape = (0,)
                else:
                    shape = (0,) + shapes[idx]

                dtype_ = dtype[idx]

                initial_maxshape = (None,) + shapes[idx] if len(shapes[idx]) != 0 else (None,)
                initial_chunkshape = (chunk_len,) + shapes[idx] if len(shapes[idx]) != 0 else (chunk_len,)
                
                self.dset = h5f.create_dataset(
                    i,
                    shape=shape,
                    maxshape=initial_maxshape, 
                    dtype=dtype_,
                    compression=compression,
                    chunks=initial_chunkshape
                )

    @contextmanager
    def open_hdf5_file(self, mode='a'):
        file = h5py.File(self.datapath, mode)
        try:
            yield file
        finally:
            file.close()

    def append(self, dataset, values, shape):
        with self.open_hdf5_file('a') as h5f:
            dset = h5f[dataset]
            new_size = (self.i[dataset] + 1,) + shape if len(shape) != 0 else (self.i[dataset] + 1,)
            dset.resize(new_size)
            if isinstance(values, str):
                values = np.array([values], dtype=h5py.string_dtype(encoding='utf-8'))
            dset[self.i[dataset]] = values
            self.i[dataset] += 1
            h5f.flush()

class VindrH5Converter(BaseH5Converter):
    def __init__(self, data_dir: str, output_dir: str, img_size: int = 224, chunk_size: int = 1000, num_processes: int = 2, split='training'):
        super().__init__(data_dir, output_dir, chunk_size, num_processes)
        
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
            return path, image
        except Exception as e:
            logging.warning(f"Failed to process {path}: {e}")
            return None

    def convert(self, split: str = 'training'):
        
        self._init_df(split)
        all_dicom_paths = self.df.index.tolist()
        total_files = len(all_dicom_paths)
        num_hdf5_files = ceil(total_files / self.chunk_size)

        logging.info(f"Total DICOM files found: {total_files}")
        logging.info(f"Target HDF5 files to create: {num_hdf5_files}")

        start_index = 0

        for i in trange(num_hdf5_files, desc="Converting chunks", unit="chunk"):
            end_index = min(start_index + self.chunk_size, total_files)
            chunk_paths = all_dicom_paths[start_index:end_index]
            hdf5_filename = os.path.join(self.output_dir, f"chunk_{i:04d}.h5")

            hdf5_store = HDF5Store(
                datapath=hdf5_filename,
                datasets=["images", "image_paths", "birads_labels", "lesions_labels"],
                shapes=[(self.img_size, self.img_size), (), (), ()],
                dtype=[np.uint8, h5py.string_dtype(encoding='utf-8'), np.uint8, np.uint8]
            )

            with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                for result in tqdm(executor.map(self._process_dicom_image, chunk_paths), total=len(chunk_paths), desc="Processing DICOMs", leave=False):
                    if result is None:
                        continue
                    path, image = result
                    try:
                        birads_label = self.birads_dict[path]
                        lesion_label = self.lesions_dict[path]
                    
                        hdf5_store.append("images", image, image.shape)
                        hdf5_store.append("image_paths", np.array(path, dtype=h5py.string_dtype(encoding='utf-8')), ())
                        hdf5_store.append("birads_labels", np.array(birads_label, dtype=np.uint8), ())
                        hdf5_store.append("lesions_labels", np.array(lesion_label, dtype=np.uint8), ())
                    except Exception as e:
                        logging.warning(f"Failed to process {path}: {e}")

            logging.info(f"HDF5 file saved: {hdf5_filename}")
            start_index = end_index