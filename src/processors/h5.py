import logging, h5py, os, math, gc, shutil, json
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from time import perf_counter
from src.utils.io import preload_to_local
from src.operations.read import read_dicom
from src.operations.transform import resize_square
from typing import List
from src.loaders.vindr import VindrDataframeLoader


class H5BatchProcessor:
    def __init__(self, batch_size: int = 500, n_workers: int = 4, tmp_dir: str = None):
        """
        Parameters:
            batch_size (int): Number of files to process per batch.
            n_workers (int): Number of parallel threads to use.
            tmp_dir (str): Temporary directory for file copying if needed.
        """
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.tmp_dir = tmp_dir

    # TODO: Extend for any dataset
    def write_h5_file(self, filename: str, image_batch: List[np.ndarray], ) -> None:
        if not image_batch:
            logging.warning(f"No images to write to {filename}")
            return
        with h5py.File(filename, 'w') as h5_file:
            h5_file.create_dataset("images", data=np.array(image_batch), compression="gzip")

    # TODO: Move to pipeline
    def read_and_resize(self, path: str, new_size: int) -> np.ndarray:
        image = read_dicom(path)
        if image is not None:
            return resize_square(image, new_size)
        return None

    def process_batch(self, dicom_file_paths: List[str], output_dir: str) -> None:
        num_chunks = math.ceil(len(dicom_file_paths) / self.batch_size)
        with tqdm(total=num_chunks, desc="Processing batches") as pbar:
            for idx in range(num_chunks):
                batch_paths = dicom_file_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
                description_prefix = f"Chunk {idx}/{num_chunks}"

                try:
                    if self.tmp_dir:
                        pbar.set_description(f"{description_prefix} - Copying to temp dir")
                        batch_paths, _ = preload_to_local(batch_paths, custom_dir=self.tmp_dir, max_files=self.batch_size)

                    pbar.set_description(f"{description_prefix} - Processing")
                    t_start = perf_counter()
                    with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                        batch_images = list(
                            tqdm(
                                executor.map(lambda path: self.read_and_resize(path, 224), batch_paths),
                                total=len(batch_paths),
                                leave=False,
                            )
                        )
                    batch_images = [img for img in batch_images if img is not None]
                    pbar.set_description(f"{description_prefix} - Saving batch to HDF5")    
                    filename = os.path.join(output_dir, f"batch_{idx:04d}.h5")
                    self.write_h5_file(filename, batch_images)
                finally:
                    if self.tmp_dir:
                        pbar.set_description(f"{description_prefix} - Cleaning up temp dir")
                        shutil.rmtree(self.tmp_dir)
                    del batch_images
                    gc.collect()

                    pbar.set_description(f"{description_prefix} - Done in {perf_counter() - t_start:.2f}s")
                    pbar.update()

    def run(self, filepath_list: List[str], output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        num_batches = math.ceil(len(filepath_list) / self.batch_size)
        logging.info(f"Total files: {len(filepath_list)}")
        logging.info(f"Processing in {num_batches} batches of {self.batch_size}")
        logging.info(f"Using {self.n_workers} workers")
        logging.info(f"Output directory: {output_dir}")
        self.process_batch(filepath_list, output_dir)
        logging.info("All batches processed successfully.")
        
        
class VindrH5BatchProcessor(H5BatchProcessor):
    def __init__(self, data_dir: str, batch_size: int = 500, n_workers: int = 4, tmp_dir: str = None):
        super().__init__(batch_size=batch_size, n_workers=n_workers, tmp_dir=tmp_dir)
        self.df_loader = VindrDataframeLoader(data_dir)
        
    def write_h5_file(self, filename: str, image_batch: List[np.ndarray], birads_batch: List[int], lesions_batch: List[int]) -> None:
        if not image_batch:
            logging.warning(f"No images to write to {filename}")
            return
        with h5py.File(filename, 'w') as h5_file:
            h5_file.create_dataset("x", data=np.array(image_batch), compression="gzip")
            birads_dataset = h5_file.create_dataset("y_birads", data=np.array(birads_batch, dtype=np.int32), compression="gzip")
            lesions_dataset = h5_file.create_dataset("y_lesions", data=np.array(lesions_batch, dtype=np.int32), compression="gzip")
            birads_dataset.attrs['label_mapping'] = json.dumps(self.df_loader.birads_mapping)
            lesions_dataset.attrs['label_mapping'] = json.dumps(self.df_loader.lesions_mapping)

    def process_batch(self, dicom_file_paths: List[str], output_dir: str) -> None:
        num_chunks = math.ceil(len(dicom_file_paths) / self.batch_size)
        with tqdm(total=num_chunks, desc="Processing batches") as pbar:
            for idx in range(num_chunks):
                batch_paths = dicom_file_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
                description_prefix = f"Chunk {idx}/{num_chunks}"
                try:
                    if self.tmp_dir:
                        pbar.set_description(f"{description_prefix} - Copying to temp dir")
                        batch_paths, _ = preload_to_local(batch_paths, custom_dir=self.tmp_dir, max_files=self.batch_size)

                    batch_birads = [self.birads_dict[path] for path in batch_paths]
                    batch_lesions = [self.lesions_dict[path] for path in batch_paths]
                    pbar.set_description(f"{description_prefix} - Processing")
                    t_start = perf_counter()
                    with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                        batch_images = list(
                            tqdm(
                                executor.map(lambda path: self.read_and_resize(path, 224), batch_paths),
                                total=len(batch_paths),
                                leave=False,
                            )
                        )
                    batch_images = [img for img in batch_images if img is not None]
                    pbar.set_description(f"{description_prefix} - Saving batch to HDF5")    
                    filename = os.path.join(output_dir, f"batch_{idx:04d}.h5")
                    self.write_h5_file(filename, batch_images, batch_birads, batch_lesions)
                finally:
                    if self.tmp_dir:
                        pbar.set_description(f"{description_prefix} - Cleaning up temp dir")
                        shutil.rmtree(self.tmp_dir)
                    del batch_images
                    gc.collect()

                    pbar.set_description(f"{description_prefix} - Done in {perf_counter() - t_start:.2f}s")
                    pbar.update()

    def run(self, output_dir: str, split: str = 'training') -> None:
        df = self.df_loader(split=split)
        df.set_index('absolute_path', inplace=True)
        paths = df.index.tolist()
        self.birads_dict = df['breast_birads'].to_dict()
        self.lesions_dict = df['finding_categories'].to_dict()

        os.makedirs(output_dir, exist_ok=True)
        num_batches = math.ceil(len(paths) / self.batch_size)
        logging.info(f"Total files: {len(paths)}")
        logging.info(f"Processing in {num_batches} batches of {self.batch_size}")
        logging.info(f"Using {self.n_workers} workers")
        logging.info(f"Output directory: {output_dir}")
        self.process_batch(paths, output_dir)
        logging.info("All batches processed successfully.")
        