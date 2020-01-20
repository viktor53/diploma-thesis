from typing import List, Callable, Dict, Iterable
import logging
from os import linesep, sep
from collections import Counter
from datetime import datetime
from constants import PATH_TO_DATA, SPLIT_FILES


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

    def transform(self, line: str, first_line: bool = False) -> List[str]:
        if first_line:
            self._header = self._extract_header_mapping(line[:-1].split(self._csv_sep))
            return [col for col, _ in self._header.items()]
        else:
            transformed = self._make_transformation(self._header, line[:-1].split(self._csv_sep))
            replaced = self._make_replacing(self._header, transformed)
            dropped = self._make_drop(self._header, replaced)

            return dropped

    def run(self,path_to_files: str, files: List[str], results_dir: str):
        for file in files:
            self._logger.info("Processing file: {}".format(file))
            with open(path_to_files + sep + file, "r") as in_file:
                with open(results_dir + sep + file, "w") as out_file:
                    first_line = True
                    for line in in_file:
                        line_to_write = self.transform(line, first_line)

                        out_file.write(self._csv_sep.join(line_to_write))
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
    for line in iterator:
        if first_line:
            first_line = False
        else:
            counter.update([line[:-1].split(",")[-1]])

    return counter


def write_line(out_file, line_to_write: List[str], csv_sep: str = ","):
    out_file.write(csv_sep.join(line_to_write))
    out_file.write(linesep)


def split_train_test(path_to_data: str, files: List[str], path_to_results: str, transformer: Transformer = None,
                     train_ratio: float = 0.8):
    with open(path_to_results + sep + "train.csv", "w") as train:
        with open(path_to_results + sep + "test.csv", "w") as test:
            header_is_written = False
            for file in files:
                logging.info("Processing file: {}".format(file))

                number_fo_classes = dict()
                with open(path_to_data + sep + file, "r") as in_file:
                    logging.info("Checking classes counts.")

                    for cls, count in get_number_of_classes_in_file(in_file).items():
                        number_fo_classes.update([(cls, int(count * train_ratio))])

                with open(path_to_data + sep + file, "r") as in_file:
                    logging.info("Splitting into train and test.")

                    first_line = True

                    if transformer is not None:
                        for line in in_file:
                            if first_line:
                                if not header_is_written:
                                    line_to_write = transformer.transform(line, first_line)
                                    write_line(train, line_to_write)
                                    write_line(test, line_to_write)

                                    header_is_written = True

                                first_line = False
                            else:
                                line_to_write = transformer.transform(line, first_line)

                                number_fo_classes[line_to_write[-1]] -= 1

                                if number_fo_classes[line_to_write[-1]] > 0:
                                    write_line(train, line_to_write)
                                else:
                                    write_line(test, line_to_write)
                    else:
                        for line in in_file:
                            if first_line:
                                if not header_is_written:
                                    line_to_write = line[-1].split(",")

                                    write_line(train, line_to_write)
                                    write_line(test, line_to_write)

                                    header_is_written = True

                                first_line = False
                            else:
                                line_to_write = line[-1].split(",")

                                number_fo_classes[line_to_write[-1]] -= 1

                                if number_fo_classes[line_to_write[-1]] > 0:
                                    write_line(train, line_to_write)
                                else:
                                    write_line(test, line_to_write)


def check_time(path_to_data: str, files: List[str]):
    for file in files:
        logging.info("Processing file: {}".format(file))
        with open(path_to_data + sep + file, "r") as in_file:
            prev_time = 0
            first_line = True
            line_number = 0
            for line in in_file:
                if first_line:
                    first_line = False
                else:
                    split_line = line[:-1].split(",")
                    current_time = float(split_line[2])

                    if current_time < prev_time:
                        logging.error("Line {} - Current time {} is smaller than previous time {}".format(line_number,
                                                                                                          current_time, prev_time))

                    prev_time = current_time

                line_number += 1


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

    # check_time("../clean_data", SPLIT_FILES)

    transformer = Transformer().set_drop(["Timestamp"])
    split_train_test(PATH_TO_DATA, SPLIT_FILES, "../prepared_data", transformer)


