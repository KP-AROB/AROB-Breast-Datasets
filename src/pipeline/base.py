import inspect
from typing import Generator


class BasePipeline(object):
    def __init__(self):
        self._operations = []

    def add_operation(self, ops: Generator):
        """Add a processing operation to the pipeline

        Args:
            ops (Generator): The processing generator

        Raises:
            TypeError: throws an error if the function passed is not a generator
        """
        if inspect.isgeneratorfunction(ops):
            self._operations.append(ops)
        else:
            raise TypeError(f"{ops.__name__} is not a Generator")

    def process(self, source: str):
        """Runs the pipeline.
        Args:
            source (str): The .dicom file path
        Returns:
            np.array: The processed data
        """
        for ops in self._operations:
            source = ops(source)
        return source
