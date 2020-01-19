from typing import List, Callable, Dict
import logging
from os import linesep, sep
from datetime import datetime
from constants import PATH_TO_DATA, SPLIT_FILES


class Transformer:

    def __init__(self, csv_sep: str = ","):
        self._replacement = []
        self._transformation = []
        self._drop = []
        self._csv_sep = csv_sep

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
        return [row[i] for _, i in header_mapping]

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

    def run(self,path_to_files: str, files: List[str], results_dir: str):
        for file in files:
            with open(path_to_files + sep + file, "r") as in_file:
                with open(results_dir + sep + file, "w") as out_file:
                    header = None
                    first_line = True
                    for line in in_file:
                        if first_line:
                            first_line = False
                            header = self._extract_header_mapping(line[:-1].split(self._csv_sep))
                            out_file.write(",".join([col for col, _ in header]))
                        else:
                            transformed = self._make_transformation(header, line[:-1].split(self._csv_sep))
                            replaced = self._make_replacing(header, transformed)
                            dropped = self._make_drop(header, replaced)

                            out_file.write(self._csv_sep.join(dropped))

                        out_file.write(linesep)


def split_files_with_more_headers(path_to_data: str, files: List[str]):
    for file in files:
        logging.info("Processing file: " + file)
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


if __name__ == "__main__":
    # datetime is converted to timestamp
    # NaN value is replaced by most common value in a column
    # Infinity value is replaced by maximum value in a column
    # columns, that have only one distinct value, are dropped
    transformer = Transformer()\
        .add_transformation("Timestamp", lambda s: str(datetime.strptime(s, "%d/%m/%Y %H:%M:%S").timestamp()))\
        .add_replacement("Flow Byts/s", "NaN", "0")\
        .add_replacement("Flow Byts/s", "Infinity", "1806642857.14286")\
        .add_replacement("Flow Pkts/s", "NaN", "1000000")\
        .add_replacement("Flow Pkts/s", "Infinity", "6000000.0")\
        .set_drop(["Bwd Blk Rate Avg", "Bwd Byts/b Avg", "Bwd PSH Flags", "Bwd Pkts/b Avg", "Bwd URG Flags",
                   "Fwd Blk Rate Avg", "Fwd Byts/b Avg", "Fwd Pkts/b Avg"])

    transformer.run(PATH_TO_DATA, SPLIT_FILES, "../clean_data")
