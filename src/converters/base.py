import logging, os

class BaseH5Converter:
    """
    Base class for HDF5 converters.
    """
    def __init__(self, 
                 data_dir: str, 
                 output_dir: str,
                 chunk_size: int = 1000,
                 num_threads: int = None):
        
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.chunk_size = chunk_size

        if chunk_size <= 0:
            raise ValueError("chunk size must be positive.")
        if num_threads is None or num_threads <= 0:
            logging.warning(f"num_threads ({num_threads}) is invalid. Using 1 process.")
            self.num_threads = 16
        elif num_threads <= 0:
            self.num_threads = 16
        else:
            self.num_threads = num_threads 

        logging.info(f"Using {self.num_threads} worker threads.")