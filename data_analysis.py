from typing import List, Dict, Callable
from os import sep, linesep
import concurrent.futures
from collections import Counter
from data import Row, Dataset
from constants import PATH_TO_DATA, SPLIT_FILES, CONF_DIR, CLASSES_ANALYSIS_CONF, ANALYSIS_BY_PARTS, CLASSES,\
    FILES_WITH_RESULTS
import logging
import yaml
from math import sqrt, isinf, isnan
from random import shuffle


class Analyzer:
    '''
    Class Analyzer represents abstract class for all
    specific analyzers. It defines processing a dataset
    row by row. It is also possible to join results of
    the same analyzer.

    It provides basic values checks:
    - _is_valid_number(number) -> bool
    - _is_valid_str(s: str)

    Inherited classes has to implement abstract methods.

    Parameters
    ----------
    column : str
        Column name where analysis should be done
    '''

    def __init__(self, column: str):
        self._column = column

    @staticmethod
    def _is_valid_number(number) -> bool:
        return number is not None and not isinf(number) and not isnan(number)

    @staticmethod
    def _is_valid_str(s: str) -> bool:
        return s is not None

    def analyze(self, row: Row):
        '''
        Analyzes a dataset row by row. It is incremental analysis.

        Parameters
        ----------
        row : Row
            A row to analyze.
        '''
        raise NotImplementedError("You have to implement this method!")

    def get_result(self) -> List[str]:
        '''
        Gets result of the analysis. It can be called during the analysis
        or after analysis.

        Returns
        -------
        result : List[str]
            It has length of 3, where first is a column name,
            second is an analyzer name and last is the result.
        '''
        raise NotImplementedError("You have to implement this method!")

    def join(self, analyzer: 'Analyzer') -> 'Analyzer':
        '''
        Joins results of two same analyzers.

        Parameters
        ----------
        analyzer : Analyzer
            The same analyzer for the same column.

        Returns
        -------
        analyzer : Analyzer
            Returns self, modified analyzer. It does not create a copy.

        Raises
        ------
        TypeError
            If the analyzer is for different column.
        '''
        raise NotImplementedError("You have to implement this method!")

    def join_multiple(self, analyzers: List['Analyzer']) -> 'Analyzer':
        '''
        Joins results of multiple same analyzers.

        Parameters
        ----------
        analyzers : List[Analyzer]
            List of the same analyzers for the same column.

        Returns
        -------
        analyzer : Analyzer
            Returns self, modified analyzer. It does not create a copy.

        Raises
        ------
        TypeError
            If the analyzers are for different column.
        '''
        raise NotImplementedError("You have to implement this method!")


class Min(Analyzer):

    def __init__(self, column: str):
        super().__init__(column)
        self._min = None

    def analyze(self, row: Row):
        current = row.get_float_by_column(self._column)

        if Analyzer._is_valid_number(current):
            if self._min is None or self._min > current:
                self._min = current

    def get_result(self) -> List[str]:
        return [self._column, "Min", str(self._min)]

    def join(self, analyzer: 'Min') -> 'Min':
        if self._column != analyzer._column:
            raise TypeError("Joining different analyzers!")

        if self._min is not None and analyzer._min is not None:
            if self._min > analyzer._min:
                self._min = analyzer._min
        elif self._min is None:
            self._min = analyzer._min

        return self

    def join_multiple(self, analyzers: List['Min']) -> 'Min':
        for analyzer in analyzers:
            if self._column != analyzer._column:
                raise TypeError("Joining different analyzers!")

            if self._min is not None and analyzer._min is not None:
                if self._min > analyzer._min:
                    self._min = analyzer._min
            elif self._min is None:
                self._min = analyzer._min

        return self


class Max(Analyzer):

    def __init__(self, column: str):
        super().__init__(column)
        self._max = None

    def analyze(self, row: Row):
        current = row.get_float_by_column(self._column)

        if Analyzer._is_valid_number(current):
            if self._max is None or self._max < current:
                self._max = current

    def get_result(self) -> List[str]:
        return [self._column, "Max", str(self._max)]

    def join(self, analyzer: 'Max') -> 'Max':
        if self._column != analyzer._column:
            raise TypeError("Joining different analyzers!")

        if self._max is not None and analyzer._max is not None:
            if self._max < analyzer._max:
                self._max = analyzer._max
        elif self._max is None:
            self._max = analyzer._max

        return self

    def join_multiple(self, analyzers: List['Max']) -> 'Max':
        for analyzer in analyzers:
            if self._column != analyzer._column:
                raise TypeError("Joining different analyzers!")

            if self._max is not None and analyzer._max is not None:
                if self._max < analyzer._max:
                    self._max = analyzer._max
            elif self._max is None:
                self._max = analyzer._max

        return self


class Mean(Analyzer):

    def __init__(self, column: str):
        super().__init__(column)
        self._sum = 0.0
        self._number_of_rows = 0

    def analyze(self, row: Row):
        current = row.get_float_by_column(self._column)

        if Analyzer._is_valid_number(current):
            self._sum += current
            self._number_of_rows += 1

    def get_result(self) -> List[str]:
        return [self._column, "Mean", str(self._sum / self._number_of_rows)]

    def join(self, analyzer: 'Mean') -> 'Mean':
        if self._column != analyzer._column:
            raise TypeError("Joining different analyzers!")

        self._sum += analyzer._sum
        self._number_of_rows += analyzer._number_of_rows

        return self

    def join_multiple(self, analyzers: List['Mean']) -> 'Mean':
        for analyzer in analyzers:
            if self._column != analyzer._column:
                raise TypeError("Joining different analyzers!")

            self._sum += analyzer._sum
            self._number_of_rows += analyzer._number_of_rows

        return self


class StandardDeviation(Analyzer):

    def __init__(self, column: str):
        super().__init__(column)
        self._sum = 0.0
        self._sum2 = 0.0
        self._number_of_rows = 0

    def analyze(self, row: Row):
        current = row.get_float_by_column(self._column)

        if Analyzer._is_valid_number(current):
            self._sum += current
            self._sum2 += pow(current, 2)
            self._number_of_rows += 1

    def get_result(self) -> List[str]:
        result = sqrt((self._sum2 / self._number_of_rows) - pow(self._sum / self._number_of_rows, 2))

        return [self._column, "StandardDeviation", str(result)]

    def join(self, analyzer: 'StandardDeviation') -> 'StandardDeviation':
        if self._column != analyzer._column:
            raise TypeError("Joining different analyzers!")

        self._sum += analyzer._sum
        self._sum2 += analyzer._sum2
        self._number_of_rows += analyzer._number_of_rows

        return self

    def join_multiple(self, analyzers: List['StandardDeviation']) -> 'StandardDeviation':
        for analyzer in analyzers:
            if self._column != analyzer._column:
                raise TypeError("Joining different analyzers!")

            self._sum += analyzer._sum
            self._sum2 += analyzer._sum2
            self._number_of_rows += analyzer._number_of_rows

        return self


class Unique(Analyzer):

    def __init__(self, column: str):
        super().__init__(column)
        self._counter = Counter()

    def analyze(self, row: Row):
        current = row.get_str_by_column(self._column)

        if Analyzer._is_valid_str(current):
            self._counter.update([current])

    def get_result(self) -> List[str]:
        result = 0

        for _, count in self._counter.items():
            if count == 1:
                result += 1

        return [self._column, "Unique", str(result)]

    def join(self, analyzer: 'Unique') -> 'Unique':
        if self._column != analyzer._column:
            raise TypeError("Joining different analyzers!")

        self._counter.update(analyzer._counter)

        return self

    def join_multiple(self, analyzers: List['Unique']) -> 'Unique':
        for analyzer in analyzers:
            if self._column != analyzer._column:
                raise TypeError("Joining different analyzers!")

            self._counter.update(analyzer._counter)

        return self


class Distinct(Analyzer):

    def __init__(self, column: str):
        super().__init__(column)
        self._set = set()

    def analyze(self, row: Row):
        current = row.get_str_by_column(self._column)

        if Analyzer._is_valid_str(current):
            self._set.update([current])

    def get_result(self) -> List[str]:
        return [self._column, "Distinct", str(len(self._set))]

    def join(self, analyzer: 'Distinct') -> 'Distinct':
        if self._column != analyzer._column:
            raise TypeError("Joining different analyzers!")

        self._set.update(analyzer._set)

        return self

    def join_multiple(self, analyzers: List['Distinct']) -> 'Distinct':
        sets = []
        for analyzer in analyzers:
            if self._column != analyzer._column:
                raise TypeError("Joining different analyzers!")

            sets.append(analyzer._set)

        self._set.update(*sets)

        return self


class Compliance(Analyzer):

    def __init__(self, column: str, name: str, condition: Callable[[str], bool]):
        super().__init__(column)
        self._name = name
        self._condition = condition
        self._number_of_satisfied = 0

    def analyze(self, row: Row):
        current = row.get_str_by_column(self._column)

        if Analyzer._is_valid_str(current) and self._condition(current):
            self._number_of_satisfied += 1

    def get_result(self) -> List[str]:
        return [self._column, self._name, str(self._number_of_satisfied)]

    def join(self, analyzer: 'Compliance') -> 'Compliance':
        self._number_of_satisfied += analyzer._number_of_satisfied

        return self

    def join_multiple(self, analyzers: List['Compliance']) -> 'Compliance':
        for analyzer in analyzers:
            if self._column != analyzer._column or self._name != analyzer._name:
                raise TypeError("Joining different analyzers!")

            self._number_of_satisfied += analyzer._number_of_satisfied

        return self


class MostFrequent(Analyzer):

    def __init__(self, column: str, n: int = 1):
        super().__init__(column)
        self._counter = Counter()
        self._n = n

    def analyze(self, row: Row):
        current = row.get_str_by_column(self._column)

        if Analyzer._is_valid_str(current):
            self._counter.update([current])

    def get_result(self) -> List[str]:
        return [self._column, "MostFrequent", "|".join([str(e) for e, _ in self._counter.most_common(self._n)])]

    def join(self, analyzer: 'MostFrequent') -> 'MostFrequent':
        if self._column != analyzer._column:
            raise TypeError("Joining different analyzers!")

        self._counter.update(analyzer._counter)

        return self

    def join_multiple(self, analyzers: List['MostFrequent']) -> 'MostFrequent':
        for analyzer in analyzers:
            if self._column != analyzer._column:
                raise TypeError("Joining different analyzers!")

            self._counter.update(analyzer._counter)

        return self


class NumberOfRows(Analyzer):

    def __init__(self):
        super().__init__("ALL")
        self._number_of_rows = 0

    def analyze(self, row: Row):
        self._number_of_rows += 1

    def get_result(self) -> List[str]:
        return [self._column, "NumberOfRows", str(self._number_of_rows)]

    def join(self, analyzer: 'NumberOfRows') -> 'NumberOfRows':
        self._number_of_rows += analyzer._number_of_rows

        return self

    def join_multiple(self, analyzers: List['NumberOfRows']) -> 'NumberOfRows':
        for analyzer in analyzers:
            self._number_of_rows += analyzer._number_of_rows

        return self


class Filter:
    '''
    Class Filter represents abstract class for all filters.
    It provides possibility to filter out not suitable rows.
    '''

    def filter(self, row: Row) -> bool:
        '''
        Filters out not suitable rows.

        Parameters
        ----------
        row : Row
            A row.

        Returns
        -------
        True if the row is not suitable otherwise False.
        '''
        raise NotImplementedError("You have to implement this method!")


class ValueFilter(Filter):
    '''
    Class ValueFilter represents filter based on a value of a column.
    All rows which do not have the value value_to_satisfy in the
    column are filtered out.

    Parameters
    ----------
    column : str
        A column name where filter should be applied.
    value_to_satisfy : str
        A value which should be satisfied to pass the filter.
    '''

    def __init__(self, column: str, value_to_satisfy: str):
        self._column = column
        self._value_to_satisfy = value_to_satisfy

    def filter(self, row: Row) -> bool:
        current = row.get_str_by_column(self._column)

        return current is None or current != self._value_to_satisfy


def get_analyzers_mapping() -> Dict[str, Callable[[str], Analyzer]]:
    '''
    Creates mapping between analyzer names and analyzer factory methods.
    (It is used to parse yaml configurations)

    Returns
    -------
    mapping : Dict[str, Callable[[str], Analyzer]
        The key is a name of analyzer and the value is a factory method
        which creates an analyzer for specific column.
    '''

    mapping = dict()
    mapping.update([("Min", lambda c: Min(c))])
    mapping.update([("Max", lambda c: Max(c))])
    mapping.update([("Mean", lambda c: Mean(c))])
    mapping.update([("Unique", lambda c: Unique(c))])
    mapping.update([("Distinct", lambda c: Distinct(c))])
    mapping.update([("MostFrequent", lambda c: MostFrequent(c))])
    mapping.update([("StandardDeviation", lambda c: StandardDeviation(c))])
    mapping.update([("ComplianceBenign", lambda c: Compliance(c, "ComplianceBenign", lambda v: v == "Benign"))])
    mapping.update([("ComplianceInfilteration", lambda c: Compliance(c, "ComplianceInfilteration", lambda v: v == "Infilteration"))])
    mapping.update([("ComplianceSQLInjection", lambda c: Compliance(c, "ComplianceSQLInjection", lambda v: v == "SQL Injection"))])
    mapping.update([("ComplianceBot", lambda c: Compliance(c, "ComplianceBot", lambda v: v == "Bot"))])
    mapping.update([("ComplianceBruteForce-Web", lambda c: Compliance(c, "ComplianceBruteForce-Web", lambda v: v == "Brute Force -Web"))])
    mapping.update([("ComplianceBruteForce-XSS", lambda c: Compliance(c, "ComplianceBruteForce-XSS", lambda v: v == "Brute Force -XSS"))])
    mapping.update([("ComplianceFTP-BruteForce", lambda c: Compliance(c, "ComplianceFTP-BruteForce", lambda v: v == "FTP-BruteForce"))])
    mapping.update([("ComplianceSSH-BruteForce", lambda c: Compliance(c, "ComplianceSSH-BruteForce", lambda v: v == "SSH-Bruteforce"))])
    mapping.update([("ComplianceDDOSAttack-HOIC", lambda c: Compliance(c, "ComplianceDDOSAttack-HOIC", lambda v: v == "DDOS attack-HOIC"))])
    mapping.update([("ComplianceDDOSAttack-LOIC-UDP", lambda c: Compliance(c, "ComplianceDDOSAttack-LOIC-UDP", lambda v: v == "DDOS attack-LOIC-UDP"))])
    mapping.update([("ComplianceDDoSAttacks-LOIC-HTTP", lambda c: Compliance(c, "ComplianceDDoSAttacks-LOIC-HTTP", lambda v: v == "DDoS attacks-LOIC-HTTP"))])
    mapping.update([("ComplianceDoSAttacks-Hulk", lambda c: Compliance(c, "ComplianceDoSAttacks-Hulk", lambda v: v == "DoS attacks-Hulk"))])
    mapping.update([("ComplianceDoSAttacks-SlowHTTPTest", lambda c: Compliance(c, "ComplianceDoSAttacks-SlowHTTPTest", lambda v: v == "DoS attacks-SlowHTTPTest"))])
    mapping.update([("ComplianceDoSAttacks-Slowloris", lambda c: Compliance(c, "ComplianceDoSAttacks-Slowloris", lambda v: v == "DoS attacks-Slowloris"))])
    mapping.update([("ComplianceDoSAttacks-GoldenEye", lambda c: Compliance(c, "ComplianceDoSAttacks-GoldenEye", lambda v: v == "DoS attacks-GoldenEye"))])

    return mapping


def load_analysis_conf(path_to_conf: str) -> Dict[str, List[str]]:
    '''
    Loads analysis configuration.

    Parameters
    ----------
    path_to_conf : str
        A path to an yaml configuration file.

    Returns
    -------
    configuration : Dict[str, List[str]]
        Loaded yaml configuration where the key is a column name
        and the value is a list of analyzers for the column.
    '''
    with open(path_to_conf, "r") as file:
        conf = yaml.load(file)

    return conf[0]


class AnalysisRunner:
    '''
    Class AnalysisRunner represents runner of analysis.
    It provides possibility to set up an analysis and
    then run it in single thread or in multiple threads.
    After analysis it stores its results.

    Parameters
    ----------
    dataset : Dataset
        A dataset to analyze
    conf : Dict[str, List[str]]
        A configuration of analysis where the key is a column name
        and the value is a list of analyzers for the column.
    analyzers_mapping : Dict[str, Callable[[str], Analyzer]]
        A mapping between analyzer names and analyzer factory methods.
        The key is a name of analyzer and the value is a factory method
        which creates an analyzer for specific column.
    filters : List[Filter]
        A list of filters
    max_analyzers : int
        To reduce a consumption of resources user can specify
        maximum analyzers to be used in a single passing of the dataset.
        All analyzers will be processed, but it will go over the dataset
        more times.
    '''

    _logger = logging.getLogger("AnalysisRunner")

    def __init__(self, dataset: Dataset, conf: Dict[str, List[str]], analyzers_mapping: Dict[str, Callable[[str], Analyzer]],
                 filters: List[Filter] = [], max_analyzers: int = 120):
        self._dataset = dataset
        self._conf = conf
        self._analyzers_mapping = analyzers_mapping
        self._max_analyzers = max_analyzers
        self._filters = filters

    def _load_analyzers(self):
        result = []
        header = self._dataset.get_header()
        for column, analyzers in self._conf.items():
            if column in header:
                for analyzer in analyzers:
                    if analyzer in self._analyzers_mapping.keys():
                        result.append(self._analyzers_mapping[analyzer](column))
                    else:
                        self._logger.warning("Unknown analyzer {}.".format(analyzer))
            else:
                self._logger.warning("Column {} is not contained in dataset.".format(column))

        result.append(NumberOfRows())

        return result

    def _store_result(self, file_with_results: str, analyzers: List[Analyzer], new: bool = False):
        self._logger.info("Storing results into file: " + file_with_results)

        if new:
            with open(file_with_results, "w") as file:
                file.write("Column,Analyzer,Result")
                file.write(linesep)

                for analyzer in analyzers:
                    file.write(",".join(analyzer.get_result()))
                    file.write(linesep)
        else:
            with open(file_with_results, "a") as file:
                for analyzer in analyzers:
                    file.write(",".join(analyzer.get_result()))
                    file.write(linesep)

        self._logger.info("Results are stored.")

    def _run(self, iterator, analyzers, thread: int = 1):
        number = 0
        for row in iterator():
            skip = False
            for f in self._filters:
                if f.filter(row):
                    skip = True
                    break

            if not skip:
                for analyzer in analyzers:
                    analyzer.analyze(row)

            number += 1

            if number % 100000 == 0:
                self._logger.info("Thread " + str(thread) + " - Current row: " + str(number))

    def run(self, file_with_results: str):
        '''
        Runs the analysis in a single thread
        and stores the result into file file_with_results.

        Parameters
        ----------
        file_with_results : str
            A file where the results should be stored.
        '''
        self._logger.info("Running analysis.")

        analyzers = self._load_analyzers()
        partitions = int(len(analyzers) / self._max_analyzers) if len(analyzers) > self._max_analyzers else 1
        rest = int((len(analyzers) % self._max_analyzers) / partitions) + 1 if partitions > 0 else 0

        new = True
        for i in range(partitions):
            self._logger.info(
                "Running {}. part from {} (contains {} analyzers)".format(i + 1, partitions, self._max_analyzers))
            self._run(self._dataset.raw_data_without_header(), analyzers[0:self._max_analyzers + rest])
            self._logger.info(
                "Completed {}. part from {} (contains {} analyzers)".format(i + 1, partitions, self._max_analyzers))

            self._store_result(file_with_results, analyzers[0:self._max_analyzers + rest], new)
            new = False
            del analyzers[0:self._max_analyzers + rest]

        self._logger.info("Analysis completed.")

    def run_parallel(self, file_with_results: str, workers: int):
        '''
        Runs the analysis in parallel and the results stores
        into file file_with_results. The number of workers
        is specified by the parameter workers.

        Parameters
        ----------
        file_with_results : str
            A file where the results should be stored.
        workers : int
            A number of workers.
        '''
        self._logger.info("Running analysis in " + str(workers) + " threads.")

        analyzers = []
        for i in range(workers):
            analyzers.append(self._load_analyzers())

        partitions = int(len(analyzers[0]) / self._max_analyzers) if len(analyzers[0]) > self._max_analyzers else 1
        rest = int((len(analyzers[0]) % self._max_analyzers) / partitions) + 1 if partitions > 0 else 0

        new = True
        for i in range(partitions):
            self._logger.info(
                "Running {}. part from {} (contains {} analyzers)".format(i + 1, partitions, self._max_analyzers))

            with concurrent.futures.ThreadPoolExecutor(workers) as executor:
                iterators = self._dataset.raw_data_without_header_parallel(workers)
                for j in range(workers):
                    executor.submit(self._run, iterators[j], analyzers[j][0:self._max_analyzers + rest], j)

            self._logger.info(
                "Completed {}. part from {} (contains {} analyzers)".format(i + 1, partitions, self._max_analyzers))

            self._logger.info("Joining results of all workers.")

            result = []
            with concurrent.futures.ThreadPoolExecutor(workers) as executor:
                for k, analyzer in enumerate(analyzers[0][0:self._max_analyzers + rest]):
                    to_join = []
                    for l in range(1, len(analyzers)):
                        to_join.append(analyzers[l][k])
                    result.append(executor.submit(analyzer.join_multiple, to_join))

            self._logger.info("Joining is completed.")

            self._store_result(file_with_results, [r.result() for r in result], new)
            new = False

            for j in range(workers):
                del analyzers[j][0:self._max_analyzers + rest]
            del result

        self._logger.info("Analysis completed.")


def join_analysis_results(files: List[str], result: str):
    '''
    Joins analysis results into one CSV file.

    Parameters
    ----------
    files : List[str]
        A list of CSV files with analysis results.
    result : str
        A final CSV file.
    '''

    header = ["Column", "Analyzer"]
    rows = []

    first = True
    for file in files:
        with open(sep.join(["statistics", file]), "r") as f:
            header.append(file[:file.index("/")])
            skip = True
            i = 0
            for line in f:
                if skip:
                    skip = False
                elif first:
                    rows.append(line[:-1].split(','))
                elif line != "":
                    rows[i].append(line[:-1].split(',')[2])
                    i += 1

        first = False

    with open(result, "w") as r:
        r.write(",".join(header))
        r.write(linesep)

        for row in rows:
            r.write(",".join(row))
            r.write(linesep)


def run_classes_analysis():
    shuffle(SPLIT_FILES)
    dataset = Dataset(PATH_TO_DATA, SPLIT_FILES)

    with open(CONF_DIR + sep + CLASSES_ANALYSIS_CONF, "r") as file:
        logging.info("Loading configuration: " + CLASSES_ANALYSIS_CONF)
        conf = yaml.load(file)

    runner = AnalysisRunner(dataset, conf[0], get_analyzers_mapping())

    runner.run_parallel("classes_results.csv", 4)


def run_analysis_by_part():
    shuffle(SPLIT_FILES)
    dataset = Dataset(PATH_TO_DATA, SPLIT_FILES)

    for i, file_conf in enumerate(ANALYSIS_BY_PARTS):
        with open(CONF_DIR + sep + file_conf, "r") as file:
            logging.info("Loading configuration: " + file_conf)
            conf = yaml.load(file)

        runner = AnalysisRunner(dataset, conf[0], get_analyzers_mapping())

        runner.run_parallel("results_{}.csv".format(i + 1), 4)


def run_analysis_by_part_for_class(cls: str):
    shuffle(SPLIT_FILES)
    dataset = Dataset(PATH_TO_DATA, SPLIT_FILES)

    for i, file_conf in enumerate(ANALYSIS_BY_PARTS):
        with open(CONF_DIR + sep + file_conf, "r") as file:
            logging.info("Loading configuration: " + file_conf)
            conf = yaml.load(file)

        runner = AnalysisRunner(dataset, conf[0], get_analyzers_mapping(), [ValueFilter("Label", cls)])

        runner.run_parallel("{}_results_{}.csv".format(cls.replace(" ", "_"), i + 1), 4)


def run_analysis_for_all_classes():
    for cls in CLASSES:
        logging.info("Running analysis for class: {}".format(cls))
        run_analysis_by_part_for_class(cls)
        logging.info("Analysis for class {} is completed.".format(cls))


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s", level=logging.INFO)

    # run_classes_analysis()

    # run_analysis_by_part()

    # run_analysis_for_all_classes()

    # join_analysis_results(FILES_WITH_RESULTS, "results.csv")
