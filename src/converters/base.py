import logging, os

class BaseH5Converter:
    """
    Base class for HDF5 converters.
    """
    def __init__(self, 
                 data_dir: str, 
                 output_dir: str,
                 chunk_size: int = 1000,
                 num_processes: int = None):
        
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.chunk_size = chunk_size

        if chunk_size <= 0:
            raise ValueError("chunk size must be positive.")
        if num_processes is None:
            self.num_processes = os.cpu_count()
        elif num_processes <= 0:
             logging.warning(f"num_processes ({num_processes}) is invalid. Using 1 process.")
             self.num_processes = 1
        else:
            self.num_processes = min(num_processes, os.cpu_count())

        logging.info(f"Using {self.num_processes} worker processes.")