import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    plot_classes_ratio("statistics/classes_results.csv")
