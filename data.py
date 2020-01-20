from typing import List, Iterable, Dict, Set, Callable
from os import sep
from functools import partial
import logging
import csv


class Row:

    def __init__(self, mapping: Dict[str, int], row: List[str]):
        self._mapping = mapping
        self._row = row

    def get_raw_row(self) -> List[str]:
        return self._row

    def get_int_by_column(self, column: str) -> int:
        if column in self._mapping:
            return int(self._row[self._mapping[column]])
        else:
            raise KeyError("Column does not exist!")

    def get_int_by_index(self, index: int) -> int:
        if index < len(self._row):
            return int(self._row[index])
        else:
            raise IndexError("Column does not exist!")

    def get_float_by_column(self, column: str) -> float:
        if column in self._mapping:
            return float(self._row[self._mapping[column]])
        else:
            raise KeyError("Column does not exist!")

    def get_float_by_index(self, index: int) -> float:
        if index < len(self._row):
            return float(self._row[index])
        else:
            raise IndexError("Column does not exist!")

    def get_str_by_column(self, column: str) -> str:
        if column in self._mapping:
            return str(self._row[self._mapping[column]])
        else:
            raise KeyError("Column does not exist!")

    def get_str_by_index(self, index: int) -> str:
        if index < len(self._row):
            return str(self._row[index])
        else:
            raise IndexError("Column does not exist!")


class Dataset:

    _logger = logging.getLogger("Dataset")

    def __init__(self, path_to_data: str, files_names: List[str], delimiter=","):
        self._path_to_data = path_to_data
        self._files_names = files_names
        self._delimiter = delimiter
        self._header = self._get_header(delimiter)

    def _get_header(self, delimiter) -> Dict[str, int]:
        header = None
        multiple_headers = False

        for file_name in self._files_names:
            self._logger.info("Checking header of file: " + file_name)
            with open(self._path_to_data + sep + file_name) as file:
                if header is None:
                    header = file.readline()[:-1].split(delimiter)
                else:
                    current_header = file.readline()[:-1].split(delimiter)
                    if header != current_header:
                        self._logger.error("Different header for one dataset! Header: " + delimiter.join(current_header))
                        multiple_headers = True

        if multiple_headers:
            self._logger.error("Multiple headers for one dataset!")
            raise ValueError("Multiple headers for one dataset!")
        elif header is None:
            return dict()
        else:
            result = dict()
            for i, column in enumerate(header):
                result.update([(column, i)])
            return result

    def _raw_data_without_header(self, files_names: List[str]) -> Iterable[Row]:
        for file_name in files_names:
            self._logger.info("Reading file: " + file_name)
            with open(self._path_to_data + sep + file_name) as file:
                skip = True
                for row in csv.reader(file, delimiter=self._delimiter):
                    if skip:
                        skip = False
                    else:
                        yield Row(self._header, row)

    def get_header(self) -> Set[str]:
        return set(self._header.keys())

    def raw_data_without_header(self) -> Callable[[], Iterable[Row]]:
        return lambda: self._raw_data_without_header(self._files_names)

    def raw_data_without_header_parallel(self, workers: int) -> List[Callable[[], Iterable[Row]]]:
        partition_size = int(len(self._files_names) / workers)

        iterators = []
        for i in range(workers - 1):
            iterators.append(partial(self._raw_data_without_header, self._files_names[i * partition_size:(i + 1) * partition_size]))
        iterators.append(partial(self._raw_data_without_header, self._files_names[(workers - 1) * partition_size:]))

        return iterators
