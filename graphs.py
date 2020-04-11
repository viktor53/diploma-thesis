import matplotlib.pyplot as plt
import math
from typing import List
from os import sep
from constants import PATH_TO_DATA, SPLIT_FILES, CLASSES, CLASSES_MAPPING
import logging
from timeit import default_timer as timer


def plot_classes_ratio(data: str):
    labels = []
    counts = []
    number_of_rows = None

    with open(data, "r") as d:
        for line in d:
            split_line = line.split(",")
            if split_line[1].startswith("Compliance"):
                labels.append(split_line[1][10:])
                counts.append(int(split_line[2]))
            elif split_line[1] == "NumberOfRows":
                number_of_rows = int(split_line[2])

    if number_of_rows is None:
        number_of_rows = sum(counts)

    legend = []
    for label, count in zip(labels, counts):
        legend.append("{:.2f} {}".format(100 * count / number_of_rows, label))

    fig, ax = plt.subplots(figsize=(14, 8))
    wedges, texts, _ = ax.pie(counts, startangle=20, autopct=lambda pct: "{:.2f}".format(pct) if pct > 1.16 else "")
    plt.legend(wedges, legend, loc="best")
    plt.axis('equal')
    plt.title("Classes Ratio")
    plt.savefig("classes_ratio.png", dpi=300)


def plot_comparison(data: str):
    plt.style.use('seaborn-whitegrid')

    with open(data, "r") as d:
        first = True

        classes = []
        x = []
        statistics = dict()
        feature = None
        i = 0
        for line in d:
            if first:
                classes = line[:-1].split(",")[2:]
                first = False
                x = [i + 1 for i in range(len(classes))]
            elif i != 6:
                split_line = line[:-1].split(",")
                statistics.update([(split_line[1], split_line[2:])])
                feature = split_line[0]
                i += 1
            else:
                means = [float(mean) if not math.isnan(float(mean)) and not math.isinf(float(mean)) else 0 for mean in statistics["Mean"]]
                stds = [float(std) if not math.isnan(float(std)) and not math.isinf(float(std)) else 0 for std in statistics["StandardDeviation"]]

                max_y = None
                min_y = None
                for m, s in zip(means, stds):
                    new_max = m + s
                    new_min = m - s
                    if max_y is None:
                        max_y = new_max
                    elif max_y < new_max:
                        max_y = new_max

                    if min_y is None:
                        min_y = new_min
                    elif min_y > new_min:
                        min_y = new_min

                plt.clf()
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.errorbar(x=x, y=means,
                            yerr=stds, fmt='.k')
                ax.scatter(x=x, y=[float(i) for i in statistics["Min"]])
                ax.scatter(x=x, y=[float(i) for i in statistics["Max"]])
                plt.title(feature)
                plt.ylim(min_y - 0.5, max_y + 0.5)
                plt.xlabel("classes")
                plt.ylabel("value")
                plt.gca().set_position((.1, .3, .6, .6))
                plt.figtext(.75, .4, "\n".join(["{} - {}".format(i + 1, cls) for i, cls in enumerate(classes)]))

                plt.savefig("graphs/" + feature.replace(" ", "_").replace("/", "_over_") + ".png", dpi=300)

                plt.close('all')

                plt.clf()
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.scatter(x=x, y=[float(i) for i in statistics["Unique"]])
                ax.scatter(x=x, y=[float(i) for i in statistics["Distinct"]])
                plt.title(feature + " Unique and Distinct")
                plt.xlabel("classes")
                plt.ylabel("value")
                plt.gca().set_position((.1, .3, .6, .6))
                plt.figtext(.75, .4, "\n".join(["{} - {}".format(i + 1, cls) for i, cls in enumerate(classes)]))

                plt.savefig("graphs/" + feature.replace(" ", "_").replace("/", "_over_") + "unique_distinct.png", dpi=300)

                plt.close('all')

                statistics = dict()
                split_line = line[:-1].split(",")
                statistics.update([(split_line[1], split_line[2:])])
                feature = split_line[0]
                i = 1


def get_number_of_class(cls: str) -> int:
    if cls in CLASSES_MAPPING.keys():
        return CLASSES_MAPPING[cls]
    else:
        return -1


def plot_boxplot(path_to_data: str, files: List[str], column: int, take_log: bool = False):
    plt.style.use('seaborn-whitegrid')
    data = []
    header = []

    for _ in range(15):
        data.append([])

    logging.info("Starting to process column {}.".format(column + 1))
    start = timer()
    for file in files:
        logging.info("Processing file - {}".format(file))
        with open(path_to_data + sep + file, "r") as in_file:
            first_file = True
            for line in in_file:
                if first_file:
                    header = line[:-1].split(",")
                    first_file = False
                else:
                    split_line = line[:-1].split(",")

                    value = float(split_line[column])

                    if take_log:
                        value = math.log(value + 1)

                    if value is not None and not math.isnan(value) and not math.isinf(value):
                        data[get_number_of_class(split_line[-1])].append(value)

    logging.info("Data are collected, starting creating graph.")

    plt.clf()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.boxplot(data)
    plt.title(header[column])
    plt.xlabel("classes")

    if take_log:
        plt.ylabel("log value")
    else:
        plt.ylabel("value")

    plt.gca().set_position((.1, .3, .6, .6))
    plt.figtext(.75, .4, "\n".join(["{} - {}".format(i + 1, cls) for i, cls in enumerate(CLASSES)]))

    if take_log:
        plt.savefig("box_plots_log/" + header[column].replace(" ", "_").replace("/", "_over_") + ".png", dpi=300)
    else:
        plt.savefig("box_plots/" + header[column].replace(" ", "_").replace("/", "_over_") + ".png", dpi=300)

    plt.close('all')

    end = timer()

    logging.info("Graph is created.")
    logging.info("Processing column {} took {}.".format(column + 1, end - start))


def plot_boxplot_for_all():
    for i in range(80):
        if i != 2 and i != 79:
            plot_boxplot(PATH_TO_DATA, SPLIT_FILES, i)


def plot_boxplot_log():
    skip = [1, 2, 3, 18, 20, 21, 22, 23, 25, 26, 32, 33, 34, 35, 45, 46, 47, 48, 49, 50, 51, 52, 57, 58, 59, 60, 61, 62, 67, 68, 79]
    for i in range(80):
        if i not in skip:
            plot_boxplot(PATH_TO_DATA, SPLIT_FILES, i, take_log=True)


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s", level=logging.INFO)

    # plot_classes_ratio("statistics/classes_results.csv")
    # plot_comparison("statistics/comparison_results.csv")
    #plot_boxplot_for_all()
    plot_boxplot_log()
