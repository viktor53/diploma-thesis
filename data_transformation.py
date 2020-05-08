from typing import List, Callable, Dict, Iterable, Tuple
import logging
import numpy as np
import csv
from os import linesep, sep, system
from collections import Counter
from datetime import datetime
from constants import PATH_TO_DATA, SPLIT_FILES, PATH_TO_CLN_DATA, PATH_TO_PRPD_DATA, PATH_TO_LOG_NORM_DATA, TRAIN, TEST,\
                      VALIDATION, FULL_TRAIN, CLASSES_MAPPING, PATH_TO_NORM_DATA, TRAIN_SIZE, FULL_TRAIN_SIZE,\
                      VALIDATION_SIZE, TEST_SIZE, SMALL_TRAIN_SIZE, ORIGINAL_FILES_WITH_SAME_HEADER
from math import isinf, isnan, floor, ceil, log
from functools import partial


class Transformer:
    '''
    Class Transformer provides transformations of dataset.
    It can drop columns, replace values or transform values.

    Before running it has to be configured and then it
    applies all defined transformations.
    '''

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
        '''
        Adds a replacement.

        Parameters
        ----------
        column : str
            A column where the replacement should be applied.
        to_replace : str
            A value that should be replaced.
        replace_by : str
            A value that should replace the original value.

        Returns
        -------
        self
        '''

        self._replacement.append((column, lambda value: replace_by if value == to_replace else value))

        return self

    def add_transformation(self, column: str, transformation: Callable[[str], str]) -> 'Transformer':
        '''
        Adds a transformation of column.

        Parameters
        ----------
        column : str
            A column to be transformed.
        transformation: Callable[[str], str]
            A transformation that gets an original value and returns a transformed value.

        Returns
        -------
        self
        '''

        self._transformation.append((column, transformation))

        return self

    def add_drop(self, column: str) -> 'Transformer':
        '''
        Adds a column that should be drop.

        Parameters
        ----------
        column : str
            A column to be drop.

        Returns
        -------
        self
        '''

        self._drop.append(column)

        return self

    def set_drop(self, columns: List[str]) -> 'Transformer':
        '''
        Sets columns that should be drop.

        Parameters
        ----------
        columns : List[str]
            Columns to be drop.

        Returns
        -------
        self
        '''

        self._drop = columns

        return self

    def transform(self, row: List[str], first_line: bool = False) -> List[str]:
        '''
        Transforms a single row.

        Parameters
        ----------
        row : List[str]
            A row to be transformed.
        first_line : bool
            If it is first line it extracts a header mapping that is
            required in the later transformation.

        Returns
        -------
        row : List[str]
            A transformed row.
        '''

        if first_line:
            self._header = self._extract_header_mapping(row)
            return [col for col, _ in self._header.items()]
        else:
            transformed = self._make_transformation(self._header, row)
            replaced = self._make_replacing(self._header, transformed)
            dropped = self._make_drop(self._header, replaced)

            return dropped

    def run(self, path_to_files: str, files: List[str], results_dir: str):
        '''
        Runs a transformation of the files and stores results into result_dir.

        Parameters
        ----------
        path_to_files : str
            A path to the directory with the CSV files.
        files : List[str]
            A list of CSV files to be transformed
        results_dir : str
            A directory where results should be stored.
        '''
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


def load_header() -> List[str]:
    '''
    Loads a header of files.

    Returns
    -------
    header : List[str]
        A header of files.
    '''

    with open(PATH_TO_DATA + sep + SPLIT_FILES[0], "r") as f:
        return f.readline()[:-1].split(",")


def split_files_with_more_headers(path_to_data: str, files: List[str]):
    '''
    Splits files with more headers that are contained in middle of files.

    Parameters
    ----------
    path_to_data : str
        A path to the directory with the CSV files.
    files : List[str]
        A list of files that should be split.
    '''

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
    '''
    Gets a number of samples for each class in a file.

    Parameters
    ----------
    iterator : Iterable[str]
        An iterator over the file.

    Returns
    -------
    counter : Counter
        A counter with a number of samples for each class in a file.
    '''

    counter = Counter()

    first_line = True
    for row in iterator:
        if first_line:
            first_line = False
        else:
            counter.update([row[-1]])

    return counter


def write_row(out_file, row_to_write: List[str], csv_del: str = ","):
    '''
    Writes a row into a file.

    Parameters
    ----------
    out_file
        A file writer.
    row_to_write : List[str]
        A row that should be written into the file.
    csv_del : str
        A CSV delimiter. Default value is comma ",".

    '''

    out_file.write(csv_del.join(row_to_write))
    out_file.write(linesep)


def split_train_test(path_to_data: str, files: List[str], path_to_results: str, transformer: Transformer = None,
                     train_ratio: float = 0.8, val: bool = False):
    '''
    Splits files to a train set and test set or train set and validation set.
    The files are split based on the train_ratio. By default the train set
    consists of 80% data and test sets consists of 20% data. The classes
    ratio is preserved.

    Parameters
    ----------
    path_to_data : str
        A path to the directory with the CSV files.
    files : List[str]
        A list of CSV files to be split.
    path_to_results : str
        A path to the directory with results.
    transformer : Transformer
        A row transformer. It is optional, by default None.
    train_ratio : float
        A train ratio. It means that test ratio is 1 - train_ratio. By default 0.8.
    val : bool
        If it is train/validation split.
    '''

    train = path_to_results + sep + FULL_TRAIN if not val else path_to_results + sep + TRAIN
    test = path_to_results + sep + TEST if not val else path_to_results + sep + VALIDATION
    with open(train, "w") as train:
        with open(test, "w") as test:
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
    '''
    Checks if samples are sorted by timestamp.

    Parameters
    ----------
    path_to_data : str
        A path to the directory with CSV files.
    files : List[str]
        A list of CSV files that should be checked.
    '''

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


def get_size(dataset: str) -> Tuple[int, int, int]:
    '''
    Computes a size of dataset.

    Parameters
    ----------
    dataset : str
        A path to the dataset.

    Returns
    -------
    size : Tuple[int, int, int]
        The first is number of samples, the second is number of features
        and the last is number of negative samples.
    '''

    negatives = 0
    size = 0
    features = 0
    i = 0

    logging.info("Getting shape of data.")

    with open(dataset, "r") as in_data:
        first_line = True
        for row in csv.reader(in_data, delimiter=','):
            if first_line:
                features = len(row) - 1
                first_line = False
            else:
                if CLASSES_MAPPING[row[-1]] == 0:
                    negatives += 1

                size += 1

                if i % 100000 == 0:
                    logging.info("Row {}".format(i))

                i += 1

    logging.info("X = ({}, {}), Y = {}, number of negative values = {}".format(size, features, size, negatives))

    return size, features, negatives


def load_dataset_as_array(dataset: str, number_of_negatives: int = None) -> Tuple[np.ndarray, np.ndarray, int, int]:
    '''
    Loads a dataset numpy array. It is possible to control
    a number of negative samples.

    Parameters
    ----------
    dataset : str
        A path to the dataset.
    number_of_negatives : int
        A number of negative samples to be loaded.

    Returns
    -------
    result : Tuple[np.ndarray, np.ndarray, int, int]
        It contains in this order the X data (samples), the Y data (labels),
        the number of samples and the number of features.
    '''

    to_be_skipped = 0

    size, features, all_negatives = get_size(dataset)

    if number_of_negatives is not None:
        to_be_skipped = all_negatives - number_of_negatives
    else:
        number_of_negatives = all_negatives

    logging.info("Initializing numpy arrays.")

    data = np.zeros((size - to_be_skipped, features), dtype=np.float)
    labels = np.zeros(size - to_be_skipped, dtype=np.int)

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

                for j in range(features):
                    value = np.float(row[j])

                    if not isinf(value) and not isnan(value):
                        data[i][j] = value
                    else:
                        logging.error("Value on line {} in column {} is inf or nan after converting.".format(i + 1, j + 1))

                if i % 100000 == 0:
                    logging.info("Row {}".format(i))

                i += 1

    logging.info("Dataset is converted into numpy array.")

    return data, labels, size - to_be_skipped, features


def convert_to_npy(path_to_data: str, type: str, path_to_result: str, number_of_negatives: int = None):
    '''
    Converts a dataset into numpy array and then stores it as memory map.

    The path to the result is constructed as:
    path_to_data + sep + type + sep + "X_{}.npy".format(type)
    path_to_data + sep + type + sep + "Y_{}.npy".format(type)

    Parameters
    ----------
    path_to_data : str
        A path to the dataset.
    type : str
        A type of the dataset - train, test, validation...
    path_to_result : str
        A path to the directory with result.
    number_of_negatives : str
        A number of negative samples to be loaded.
    '''

    logging.info("Converting dataset into numpy array.")

    X, Y, size, features = load_dataset_as_array(path_to_data + sep + type + ".csv", number_of_negatives)

    logging.info("Storing X into memmap X.npy.")

    X_memmap = np.memmap(path_to_result + sep + type + sep + "X_{}.npy".format(type), dtype=np.float32, mode="write", shape=(size, features))
    X_memmap[:] = X[:]
    del X_memmap

    del X

    logging.info("Storing X is completed.")
    logging.info("Storing Y into memmap Y.npy.")

    Y_memmap = np.memmap(path_to_result + sep + type + sep + "Y_{}.npy".format(type), dtype=np.int, mode="write", shape=size)
    Y_memmap[:] = Y[:]
    del Y_memmap

    del Y

    logging.info("Storing Y is completed.")


def create_labels_for_each_class_separately(path_to_data: str, type: str, size: int):
    '''
    Converts multiclass classification into binary classification.
    It means that for each class it creates new labels that contains
    only {1, 0} where 1 is the current class and 0 is the rest.

    The path to the data is constructed as:
    path_to_data + sep + type + sep + "Y_{}.npy".format(type)

    The path to the result is constructed as:
    path_to_data + sep + type + sep + "Y_{}_cls_{}.npy".format(type, class)

    Parameters
    ----------
    path_to_data : str
        A path to the directory with data (e.g. normalized_data).
    type : str
        A type of the data (e.g. train or test).
    size : int
        A size of the dataset.
    '''

    for cls in range(0, 15):
        logging.info("Processing class: {}".format(cls))
        y_old = np.memmap(path_to_data + sep + type + sep + "Y_{}.npy".format(type), dtype=np.int, mode="r", shape=size)
        y_new = np.memmap(path_to_data + sep + type + sep + "Y_{}_cls_{}.npy".format(type, cls), dtype=np.int, mode="write", shape=size)

        for i in range(size):
            if y_old[i] == cls:
                y_new[i] = 1
            else:
                y_new[i] = 0

        del y_new

        logging.info("Processing is completed.")


def normalize(value: float, mean: float, std: float, digit: int = 4) -> float:
    '''
    Makes standardization with specific precision.

    Parameters
    ----------
    value : float
        A value to be normalized.
    mean : float
        A mean of values.
    std : float
        A standard deviation of values.
    digit : int
        A number of digits after floating point.

    Returns
    -------
    results : float
        A normalized value by standardization.
    '''

    return round((value - mean) / std, digit)


def load_normalizer() -> Transformer:
    '''
    Loads a normalizer that does standardization
    based on the train statistics.

    Returns
    -------
    normalizer : Transformer
        A loaded normalizer that does standardization.
    '''

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


def load_log_normalizer() -> Transformer:
    '''
    Loads a normalizer that takes logarithm of the data.

    Returns
    -------
    normalizer : Transformer
        A normalizer that takes logarithm of the data.
    '''

    def normalize_str(value: str):
        return str(log(float(value) + 1))

    normalizer = Transformer()

    skip = [1, 2, 3, 18, 20, 21, 22, 23, 25, 26, 32, 33, 34, 35, 45, 46, 47, 48, 49, 50, 51, 52, 57, 58, 59, 60, 61, 62,
            67, 68, 79]

    for index, column in enumerate(load_header()):
        if index not in skip:
            normalizer.add_transformation(column, normalize_str)

    return normalizer


def select_samples(path_to_data: str, type: str, ratio: float, size: int, number_of_negatives: int = 1000000) -> int:
    '''
    Selects only part of the data based on specified ratio and number of negative samples.

    Parameters
    ----------
    path_to_data : str
        A path to the directory with data.
    type : str
        A type of data (e.g. train or test)
    ratio : float
        How many percent of data should be selected with preserving
        classes ratio if it is possible. At least 10 samples for
        each class is preserved.
    size : int
        A number of samples.
    number_of_negatives : int
        A number of negative samples that is loaded before selecting.

    Returns
    -------
    result : int
        A final number of samples.
    '''

    y_old = np.memmap(path_to_data + sep + type + sep + "Y_{}.npy".format(type), dtype=np.int, mode="r", shape=size)

    logging.info("Counting classes.")

    cls_counts = Counter()
    for cls in y_old:
        cls_counts.update([cls])

    cls_steps = dict()
    cntr_steps = dict()
    for cls in range(15):
        if cls == 0 and cls_counts[0] >= number_of_negatives:
            final_counts = ceil(number_of_negatives * ratio)
        else:
            final_counts = ceil(cls_counts[cls] * ratio)

        final_counts = 10 if final_counts < 10 else final_counts
        cls_steps[cls] = floor(cls_counts[cls] / final_counts)
        cls_counts[cls] = int(final_counts)
        cntr_steps[cls] = 0

    x_old = np.memmap(path_to_data + sep + type + sep + "X_{}.npy".format(type), dtype=np.float32, mode="r", shape=(size, 70))

    new_size = sum(cls_counts.values())

    logging.info("New size of dataset is: {}".format(new_size))
    logging.info("Ratio classes is (in order 0,1,2...): {}".format(",".join([str(x) for _, x in cls_counts.items()])))

    y_new = np.memmap(path_to_data + sep + type + sep + "Y_smaller_{}.npy".format(type), dtype=np.int, mode="write", shape=new_size)
    x_new = np.memmap(path_to_data + sep + type + sep + "X_smaller_{}.npy".format(type), dtype=np.float32, mode="write", shape=(new_size, 70))

    logging.info("Creating dataset.")

    i_new = 0
    for i in range(size):
        if cntr_steps[y_old[i]] % cls_steps[y_old[i]] == 0 and cntr_steps[y_old[i]] < cls_counts[y_old[i]]:
            y_new[i_new] = y_old[i]
            x_new[i_new] = x_old[i]

            i_new += 1

        cntr_steps[y_old[i]] += 1

    logging.info("Dataset is completed.")

    return new_size


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s", level=logging.INFO)

    # removing headers in middle of file (preparation for data analysis)
    # split_files_with_more_headers(PATH_TO_DATA, ORIGINAL_FILES_WITH_SAME_HEADER[0])
    # transformer = Transformer()\
    #         .add_drop("Flow ID")\
    #         .add_drop("Src IP")\
    #         .add_drop("Src Port")\
    #         .add_drop("Dst IP")
    # transformer.run(PATH_TO_DATA, ORIGINAL_FILES_WITH_SAME_HEADER[1], "../")
    # system("mv ../" + ORIGINAL_FILES_WITH_SAME_HEADER[1][0] + " " + PATH_TO_DATA + sep + ORIGINAL_FILES_WITH_SAME_HEADER[1][0].replace(".csv", "_dropped.csv"))
    # split_files_with_more_headers(PATH_TO_DATA, [ORIGINAL_FILES_WITH_SAME_HEADER[1][0].replace(".csv", "_dropped.csv")])


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
    # transformer.run(PATH_TO_DATA, SPLIT_FILES, PATH_TO_CLN_DATA)

    # check_time(PATH_TO_CLN_DATA, SPLIT_FILES)

    # dropping Timestamp and spit to full train and test
    # transformer = Transformer().set_drop(["Timestamp"])
    # split_train_test(PATH_TO_CLN_DATA, SPLIT_FILES, PATH_TO_PRPD_DATA, transformer)

    # normalizing data by subtracting mean and dividing by std
    # normalizer = load_normalizer()
    # normalizer.run(PATH_TO_PRPD_DATA, [FULL_TRAIN, TEST], PATH_TO_NORM_DATA)

    # normalizing data by log
    # normalizer = load_log_normalizer()
    # normalizer.run(PATH_TO_PRPD_DATA, [FULL_TRAIN, TEST], PATH_TO_LOG_NORM_DATA)

    # split to train and validation
    # split_train_test(PATH_TO_NORM_DATA, [FULL_TRAIN], PATH_TO_NORM_DATA, val=True)
    # split_train_test(PATH_TO_LOG_NORM_DATA, [FULL_TRAIN], PATH_TO_LOG_NORM_DATA, val=True)

    # converting normalized data to numpy array (memmap)
    # convert_to_npy(PATH_TO_NORM_DATA, "train", PATH_TO_NORM_DATA + sep + "train")
    # convert_to_npy(PATH_TO_NORM_DATA, "full_train", PATH_TO_NORM_DATA + sep + "full_train")
    # convert_to_npy(PATH_TO_NORM_DATA, "test", PATH_TO_NORM_DATA + sep + "test")
    # convert_to_npy(PATH_TO_NORM_DATA, "validation", PATH_TO_NORM_DATA + sep + "validation")
    # convert_to_npy(PATH_TO_NORM_DATA, "train", PATH_TO_NORM_DATA, 600000) # for visualization -> size 2798546
    # create_labels_for_each_class_separately(PATH_TO_NORM_DATA, "train", TRAIN_SIZE)
    # create_labels_for_each_class_separately(PATH_TO_NORM_DATA, "full_train", FULL_TRAIN_SIZE)
    # create_labels_for_each_class_separately(PATH_TO_NORM_DATA, "validation", VALIDATION_SIZE)
    # create_labels_for_each_class_separately(PATH_TO_NORM_DATA, "test", TEST_SIZE)
    # convert_to_npy(PATH_TO_NORM_DATA, "train", PATH_TO_NORM_DATA + sep + "small_train", 1000000) # for training SVM
    # create_labels_for_each_class_separately(PATH_TO_NORM_DATA, "small_train", SMALL_TRAIN_SIZE) # for training SVM
    # size = select_samples(PATH_TO_NORM_DATA, "small_train", 0.3, SMALL_TRAIN_SIZE, number_of_negatives=1000000) # for training SVM
    # create_labels_for_each_class_separately(PATH_TO_NORM_DATA, "small_train", size) # for training SVM

    # converting log normalized data to numpy array (memmap)
    # convert_to_npy(PATH_TO_LOG_NORM_DATA, "train", PATH_TO_LOG_NORM_DATA + sep + "train")
    # convert_to_npy(PATH_TO_LOG_NORM_DATA, "full_train", PATH_TO_LOG_NORM_DATA + sep + "full_train")
    # convert_to_npy(PATH_TO_LOG_NORM_DATA, "test", PATH_TO_LOG_NORM_DATA + sep + "test")
    # convert_to_npy(PATH_TO_LOG_NORM_DATA, "validation", PATH_TO_LOG_NORM_DATA + sep + "validation")
    # create_labels_for_each_class_separately(PATH_TO_LOG_NORM_DATA, "train", TRAIN_SIZE)
    # create_labels_for_each_class_separately(PATH_TO_LOG_NORM_DATA, "full_train", FULL_TRAIN_SIZE)
    # create_labels_for_each_class_separately(PATH_TO_LOG_NORM_DATA, "validation", VALIDATION_SIZE)
    # create_labels_for_each_class_separately(PATH_TO_LOG_NORM_DATA, "test", TEST_SIZE)
