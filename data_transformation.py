from typing import List, Callable, Dict, Iterable, Tuple
import logging
import numpy as np
import csv
from os import linesep, sep
from collections import Counter
from datetime import datetime
from constants import PATH_TO_DATA, SPLIT_FILES, PATH_TO_CLN_DATA, PATH_TO_PRPD_DATA, TRAIN, TEST, CLASSES_MAPPING, PATH_TO_NORM_DATA
from math import isinf, isnan
from functools import partial


class Transformer:

    _logger = logging.getLogger("Transformer")

    def __init__(self, csv_sep: str = ","):
        self._replacement = []
        self._transformation = []
        self._drop = []
        self._csv_sep = csv_sep
        self._header = None

    def _extract_header_mapping(self, header: List[str]) -> Dict[str, int]:
        header_mapping = dict()

        for i, column in enumerate(header):
            if column not in self._drop:
                header_mapping.update([(column, i)])

        return header_mapping

    def _make_transformation(self, header_mapping: Dict[str, int], row: List[str]) -> List[str]:
        for col, trans in self._transformation:
            row[header_mapping[col]] = trans(row[header_mapping[col]])

        return row

    def _make_replacing(self, header_mapping: Dict[str, int], row: List[str]) -> List[str]:
        for col, rplc in self._replacement:
            row[header_mapping[col]] = rplc(row[header_mapping[col]])

        return row

    def _make_drop(self, header_mapping: Dict[str, int], row: List[str]) -> List[str]:
        return [row[i] for _, i in header_mapping.items()]

    def add_replacement(self, column: str, to_replace: str, replace_by: str) -> 'Transformer':
        self._replacement.append((column, lambda value: replace_by if value == to_replace else value))

        return self

    def add_transformation(self, column: str, transformation: Callable[[str], str]) -> 'Transformer':
        self._transformation.append((column, transformation))

        return self

    def add_drop(self, column: str) -> 'Transformer':
        self._drop.append(column)

        return self

    def set_drop(self, columns: List[str]) -> 'Transformer':
        self._drop = columns

        return self

    def transform(self, row: List[str], first_line: bool = False) -> List[str]:
        if first_line:
            self._header = self._extract_header_mapping(row)
            return [col for col, _ in self._header.items()]
        else:
            transformed = self._make_transformation(self._header, row)
            replaced = self._make_replacing(self._header, transformed)
            dropped = self._make_drop(self._header, replaced)

            return dropped

    def run(self, path_to_files: str, files: List[str], results_dir: str):
        for file in files:
            self._logger.info("Processing file: {}".format(file))
            with open(path_to_files + sep + file, "r") as in_file:
                with open(results_dir + sep + file, "w") as out_file:
                    first_line = True
                    for row in csv.reader(in_file, delimiter=self._csv_sep):
                        row_to_write = self.transform(row, first_line)

                        out_file.write(self._csv_sep.join(row_to_write))
                        out_file.write(linesep)

                        if first_line:
                            first_line = False


def split_files_with_more_headers(path_to_data: str, files: List[str]):
    for file in files:
        logging.info("Processing file: {}".format(file))
        index = 0
        with open(path_to_data + sep + file, "r") as original:
            header = None
            split = None
            for line in original:
                if header is None:
                    header = line
                    split = open(path_to_data + sep + file.replace(".csv", "_" + str(index) + ".csv"), "w")
                    split.write(line)
                elif header == line:
                    split.flush()
                    split.close()
                    index += 1
                    split = open(path_to_data + sep + file.replace(".csv", "_" + str(index) + ".csv"), "w")
                    split.write(line)
                else:
                    split.write(line)
            split.flush()
            split.close()


def get_number_of_classes_in_file(iterator: Iterable[str]) -> Counter:
    counter = Counter()

    first_line = True
    for row in iterator:
        if first_line:
            first_line = False
        else:
            counter.update([row[-1]])

    return counter


def write_row(out_file, row_to_write: List[str], csv_sep: str = ","):
    out_file.write(csv_sep.join(row_to_write))
    out_file.write(linesep)


def split_train_test(path_to_data: str, files: List[str], path_to_results: str, transformer: Transformer = None,
                     train_ratio: float = 0.8):
    with open(path_to_results + sep + TRAIN, "w") as train:
        with open(path_to_results + sep + TEST, "w") as test:
            header_is_written = False
            for file in files:
                logging.info("Processing file: {}".format(file))

                number_fo_classes = dict()
                with open(path_to_data + sep + file, "r") as in_file:
                    logging.info("Checking classes counts.")

                    for cls, count in get_number_of_classes_in_file(csv.reader(in_file, delimiter=',')).items():
                        number_fo_classes.update([(cls, int(count * train_ratio))])

                with open(path_to_data + sep + file, "r") as in_file:
                    logging.info("Splitting into train and test.")

                    first_line = True

                    if transformer is not None:
                        for row in csv.reader(in_file, delimiter=','):
                            if first_line:
                                if not header_is_written:
                                    row_to_write = transformer.transform(row, first_line)
                                    write_row(train, row_to_write)
                                    write_row(test, row_to_write)

                                    header_is_written = True

                                first_line = False
                            else:
                                row_to_write = transformer.transform(row, first_line)

                                number_fo_classes[row_to_write[-1]] -= 1

                                if number_fo_classes[row_to_write[-1]] > 0:
                                    write_row(train, row_to_write)
                                else:
                                    write_row(test, row_to_write)
                    else:
                        for row in csv.reader(in_file, delimiter=','):
                            if first_line:
                                if not header_is_written:
                                    write_row(train, row)
                                    write_row(test, row)

                                    header_is_written = True

                                first_line = False
                            else:
                                number_fo_classes[row[-1]] -= 1

                                if number_fo_classes[row[-1]] > 0:
                                    write_row(train, row)
                                else:
                                    write_row(test, row)


def check_time(path_to_data: str, files: List[str]):
    for file in files:
        logging.info("Processing file: {}".format(file))
        with open(path_to_data + sep + file, "r") as in_file:
            prev_time = 0
            first_line = True
            line_number = 0
            for row in csv.reader(in_file, delimiter=','):
                if first_line:
                    first_line = False
                else:
                    current_time = float(row[2])

                    if current_time < prev_time:
                        logging.error("Line {} - Current time {} is smaller than previous time {}".format(line_number,
                                                                                                          current_time, prev_time))

                    prev_time = current_time

                line_number += 1


def load_dataset_as_array(dataset: str, number_of_negatives: int = None) -> Tuple[np.ndarray, np.ndarray]:
    all_negatives = 10787690
    to_be_skipped = 0

    if number_of_negatives is not None:
        to_be_skipped = all_negatives - number_of_negatives
    else:
        number_of_negatives = all_negatives

    logging.info("Initializing numpy arrays.")

    data = np.zeros((12986236 - to_be_skipped, 70), dtype=np.float)
    labels = np.zeros(12986236 - to_be_skipped, dtype=np.int)

    logging.info("Starting to read the dataset.")

    step = int(all_negatives / number_of_negatives)
    negatives = 0
    with open(dataset, "r") as in_data:
        first_line = True
        i = 0
        for row in csv.reader(in_data, delimiter=','):
            if first_line:
                first_line = False
            else:
                if CLASSES_MAPPING[row[-1]] == 0 and (negatives % step > 0 or negatives / step >= number_of_negatives):
                    negatives += 1
                    continue

                labels[i] = CLASSES_MAPPING[row[-1]]

                if CLASSES_MAPPING[row[-1]] == 0:
                    negatives += 1

                for j in range(70):
                    value = np.float(row[j])

                    if not isinf(value) and not isnan(value):
                        data[i][j] = value
                    else:
                        logging.error("Value on line {} in column {} is inf or nan after converting.".format(i + 1, j + 1))

                if i % 100000 == 0:
                    logging.info("Row {}".format(i))

                i += 1

    logging.info("Dataset is converted into numpy array.")

    return data, labels


def convert_to_npy(dataset: str, path_to_result: str, number_of_negatives: int = None):
    all_negatives = 10787690
    to_be_skipped = 0

    if number_of_negatives is not None:
        to_be_skipped = all_negatives - number_of_negatives

    logging.info("Converting dataset into numpy array.")

    X, Y = load_dataset_as_array(dataset, number_of_negatives)

    logging.info("Storing X into memmap X.npy.")

    X_memmap = np.memmap(path_to_result + sep + "X.npy", dtype=np.float32, mode="write", shape=(12986236 - to_be_skipped, 70))
    X_memmap[:] = X[:]
    del X_memmap

    del X

    logging.info("Storing X is completed.")
    logging.info("Storing Y into memmap Y.npy.")

    Y_memmap = np.memmap(path_to_result + sep + "Y.npy", dtype=np.int, mode="write", shape=12986236 - to_be_skipped)
    Y_memmap[:] = Y[:]
    del Y_memmap

    del Y

    logging.info("Storing Y is completed.")


def normalize(value: float, mean: float, std: float, digit=4) -> float:
    return round((value - mean) / std, digit)


def load_normalizer() -> Transformer:
    def normalize_str(value: str, mean: float, std: float) -> str:
        return str(normalize(float(value), mean, std))

    normalizer = Transformer()

    with open("statistics/train/train_results.csv", mode="r") as stat:
        column_stats = dict()
        first_row = True
        for row in csv.reader(stat, delimiter=","):
            if first_row:
                first_row = False
            else:
                if row[1] == "Mean":
                    column_stats.update([("Mean", float(row[2]))])
                elif row[1] == "StandardDeviation":
                    column_stats.update([("StandardDeviation", float(row[2]))])

                if len(column_stats) == 2:
                    normalizer.add_transformation(row[0], partial(normalize_str, mean=column_stats["Mean"], std=column_stats["StandardDeviation"]))
                    column_stats = dict()

    return normalizer


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s", level=logging.INFO)

    # datetime is converted to timestamp
    # NaN value is replaced by most common value in a column
    # Infinity value is replaced by maximum value in a column
    # columns, that have only one distinct value, are dropped
    # transformer = Transformer()\
    #     .add_transformation("Timestamp", lambda s: str(datetime.strptime(s, "%d/%m/%Y %H:%M:%S").timestamp()))\
    #     .add_replacement("Flow Byts/s", "NaN", "0")\
    #     .add_replacement("Flow Byts/s", "Infinity", "1806642857.14286")\
    #     .add_replacement("Flow Pkts/s", "NaN", "1000000")\
    #     .add_replacement("Flow Pkts/s", "Infinity", "6000000.0")\
    #     .set_drop(["Bwd Blk Rate Avg", "Bwd Byts/b Avg", "Bwd PSH Flags", "Bwd Pkts/b Avg", "Bwd URG Flags",
    #                "Fwd Blk Rate Avg", "Fwd Byts/b Avg", "Fwd Pkts/b Avg"])
    #
    # transformer.run(PATH_TO_DATA, SPLIT_FILES, "../clean_data")

    # check_time(PATH_TO_CLN_DATA, SPLIT_FILES)

    # droping Timestamp
    # transformer = Transformer().set_drop(["Timestamp"])
    # split_train_test(PATH_TO_CLN_DATA, SPLIT_FILES, PATH_TO_PRPD_DATA, transformer)

    # normalizing data by subtracting mean and dividing by std
    # normalizer = load_normalizer()
    # normalizer.run(PATH_TO_PRPD_DATA, [TRAIN, TEST], PATH_TO_NORM_DATA)

    # convert_to_npy(PATH_TO_NORM_DATA + sep + TRAIN, PATH_TO_NORM_DATA)

    # convert_to_npy(PATH_TO_NORM_DATA + sep + TRAIN, PATH_TO_NORM_DATA, 600000)

