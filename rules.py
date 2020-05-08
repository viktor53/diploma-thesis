import numpy as np
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score
from skrules import SkopeRules
from model_training import load_train_npy_cls, load_validation_npy_cls, load_full_train_npy_cls, load_test_npy_cls, \
    load_header, plot_binary_confusion_matrix, load_test_npy, plot_confusion_matrix
from typing import List, Callable, Tuple
from joblib import dump, load
from timeit import default_timer as timer
import logging
import csv
import pandas as pd
import json


def duplicate_pos_class(X_data: np.memmap, Y_data: np.memmap, times: int) -> Tuple[np.ndarray, np.ndarray]:
    duplications = []

    for y in Y_data:
        if y == 1:
            duplications.append(times)
        else:
            duplications.append(1)

    Y_new_data = np.repeat(Y_data, duplications)
    X_new_data = np.repeat(X_data, duplications, axis=0)

    return X_new_data, Y_new_data


def train_skope_rules(cls: int, features_to_use: List[int], params: List[int], dup: int = 1) -> SkopeRules:
    features = np.array(features_to_use)

    header = np.array(load_header())

    clf = SkopeRules(max_depth_duplication=params[0], max_depth=params[1], n_estimators=params[2],
                     precision_min=params[3], recall_min=params[4], feature_names=header[features], verbose=3,
                     random_state=42, n_jobs=2)

    logging.info("Starting training of SkopeRules with params {}".format(params))
    start = timer()

    X_train, Y_train = load_full_train_npy_cls(cls)

    if dup > 1:
        X_train, Y_train = duplicate_pos_class(X_train[:, features], Y_train, dup)
    else:
        X_train = X_train[:, features]

    clf.fit(X_train, Y_train)

    end = timer()
    logging.info("Training is done. (Took {:.2f} minutes)".format((end - start) / 60.))

    dump(clf, "../rules/skope_rules_cls_{}.joblib".format(cls))

    for rule in clf.rules_:
        print(rule)

    return clf


def load_skope_rules(cls: int) -> SkopeRules:
    logging.info("Loading SkopeRules.")

    clf = load("../rules/skope_rules_cls_{}.joblib".format(cls))

    logging.info("Done.")

    for i, rule in enumerate(clf.rules_):
        logging.info("Rule {}: {}".format(i, str(rule)))

    return clf


def test_skope_rules(clf: SkopeRules, cls: int, features_to_use: List[int]):
    features = np.array(features_to_use)

    logging.info("Testing...")
    start = timer()

    X_test, Y_test = load_test_npy_cls(cls)
    Y_pred = clf.predict(X_test[:, features])

    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    acc = accuracy_score(Y_test, Y_pred)

    end = timer()
    logging.info(
        "Accuracy - {:.2f}, F1 Score - {:.2f}, Recall - {:.2f} (Took {:.2f} minutes)".format(acc * 100, f1, recall,
                                                                                             (end - start) / 60.))

    plot_binary_confusion_matrix(Y_pred, lambda: load_test_npy_cls(cls), cls, acc, f1, recall)


def denormalize_rules():
    '''
    Denormalizes values in rules. First it loads train statistics
    and then goes through each rule and prints the denormalized form.
    '''

    column_mean_std = dict()
    with open("../statistics/train/train_results.csv", mode="r") as stat:
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
                    column_mean_std.update([(row[0], (column_stats["Mean"], column_stats["StandardDeviation"]))])

    for cls in range(15):
        print('******* cls {} *******'.format(cls))
        clf = load_skope_rules(cls)
        for rule_pair in clf.rules_:
            rule_split = rule_pair[0].split(' ')

            feature = []
            for i in range(len(rule_split)):
                if rule_split[i].replace('.', '', 1).replace('-', '', 1).isdigit():
                    feature_name = " ".join(feature)
                    value = float(rule_split[i])
                    value = value * column_mean_std[feature_name][0] + column_mean_std[feature_name][1]
                    rule_split[i] = str(value)
                    feature = []
                elif rule_split[i] != '<=' and rule_split[i] != '>' and rule_split[i] != 'and':
                    feature.append(rule_split[i])

            print(' '.join(rule_split))

        print('**********************')


def test_multiclass_classification(sorted_by: int = 0, from_scratch: bool = True):
    '''
    Tests multiclass classification. First it loads rules
    and sorts them by one of the performance measures and
    then it runs classification.

    Parameters
    ----------
    sorted_by : int
        By default 0 - Precision. Other options are 1 - Recall and 2 - OOB.
    from_scratch : bool
        It stores the rules as JSON file, so after second file it
        does not have to load the rules from Skope-Rules model.
    '''

    header = np.array(load_header())
    new_header = ["__C__{}".format(i) for i in range(70)]
    header_mapping = dict()
    header_mapping.update(list(zip(header, new_header)))

    logging.info("Preparing rules...")

    if from_scratch:
        rules = []

        for cls in range(1, 15):
            clf = load_skope_rules(cls)

            for rule in clf.rules_:
                modified_rule = rule[0]
                for old, new in header_mapping.items():
                    modified_rule = modified_rule.replace(old, new)
                rules.append((cls, modified_rule, rule[1]))

        with open("rules_without_benign.json", "w") as f:
            json.dump(rules, f)
    else:
        with open("rules_without_benign.json", "r") as f:
            rules = json.load(f)

    if sorted_by != 2:
        sorted_rules = sorted(rules, key=lambda x: -x[2][sorted_by])
    else:
        sorted_rules = sorted(rules, key=lambda x: x[2][sorted_by])

    X, Y_true = load_test_npy()

    df = pd.DataFrame(X, columns=new_header)
    Y_pred = np.zeros(X.shape[0])

    logging.info("Predicting...")

    for rule in sorted_rules:
        indexes = df.query(rule[1]).index
        for index in indexes:
            if Y_pred[index] == 0:
                Y_pred[index] = rule[0]

    logging.info("Computing accuracy...")

    acc = accuracy_score(Y_true, Y_pred)
    f1 = f1_score(Y_true, Y_pred, average='micro')
    rec = recall_score(Y_true, Y_pred, average='micro')
    prec = precision_score(Y_true, Y_pred, average='micro')

    logging.info("Accuracy: {}, F1 Score: {}, Recall: {}, Precision: {}".format(acc, f1, rec, prec))

    plot_confusion_matrix(Y_pred, load_test_npy, acc, f1, rec)


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s", level=logging.INFO)

    # parameters: max_depth_duplication, max_depth, n_estimators, precision_min, recall_min
    # train_skope_rules(0, [69, 21, 28, 27, 45, 63, 16, 7, 36, 26, 47, 29, 30, 22, 59, 35, 33, 2, 53, 23, 13, 61, 58, 0, 55],
    #                   [3, 3, 30, 0.3, 0.1])
    # train_skope_rules(1, [13, 0], [3, 3, 30, 0.3, 0.1])
    # train_skope_rules(2, [25, 16, 50, 35, 27, 36, 19, 0, 58], [3, 7, 30, 0.3, 0.1], dup=1000)
    # train_skope_rules(3, [24, 20, 11, 69, 59, 25, 60, 33, 23, 14, 30, 52, 19, 15, 26, 61, 58, 29, 0, 36],
    #                   [3, 10, 30, 0.3, 0.1], dup=1000)
    # train_skope_rules(4, [46, 19, 35, 58, 0], [3, 3, 30, 0.3, 0.1])
    # train_skope_rules(5, [69, 24, 23], [3, 3, 30, 0.3, 0.1], dup=200)
    # train_skope_rules(6, [58, 0, 20, 2, 14], [3, 10, 40, 0.2, 0.1])
    # train_skope_rules(7, [41, 5, 38, 57, 61, 28, 58, 17, 20, 14], [3, 5, 40, 0.3, 0.1])
    # train_skope_rules(8, [7, 0, 14, 57, 15, 25, 61, 4], [3, 3, 40, 0.3, 0.1])
    # train_skope_rules(9, [28, 29, 58, 0, 36, 16, 35, 61], [3, 3, 40, 0.3, 0.1])
    # train_skope_rules(10, [9, 15, 65, 18, 41, 33, 60, 2, 40, 56, 35, 14, 11, 51, 58, 31, 69, 22, 25, 0, 39, 61, 29],
    #                   [3, 5, 40, 0.3, 0.1])
    # train_skope_rules(11, [36, 16, 61], [3, 3, 30, 0.3, 0.1])
    # train_skope_rules(12, [63, 3, 4, 68, 56, 66, 67, 57, 64, 50, 62, 47, 69, 34, 55, 65, 45, 37, 5, 8, 25, 18, 23, 60],
    #                   [5, 10, 40, 0.1, 0.1]) # not successful
    # train_skope_rules(13, [51, 21, 49, 58, 61, 0, 18, 2, 35, 14], [3, 5, 30, 0.3, 0.1], dup=10000)
    # train_skope_rules(14, [47, 8, 36, 33, 0, 34], [3, 3, 30, 0.3, 0.1])

    # running tests
    # clf = load_skope_rules(0)
    # test_skope_rules(clf, 0, [69, 21, 28, 27, 45, 63, 16, 7, 36, 26, 47, 29, 30, 22, 59, 35, 33, 2, 53, 23, 13, 61, 58, 0, 55])
    #
    # clf = load_skope_rules(1)
    # test_skope_rules(clf, 1, [13, 0])
    #
    # clf = load_skope_rules(2)
    # test_skope_rules(clf, 2, [25, 16, 50, 35, 27, 36, 19, 0, 58])
    #
    # clf = load_skope_rules(3)
    # test_skope_rules(clf, 3, [24, 20, 11, 69, 59, 25, 60, 33, 23, 14, 30, 52, 19, 15, 26, 61, 58, 29, 0, 36])
    #
    # clf = load_skope_rules(4)
    # test_skope_rules(clf, 4, [46, 19, 35, 58, 0])
    #
    # clf = load_skope_rules(5)
    # test_skope_rules(clf, 5, [69, 24, 23])
    #
    # clf = load_skope_rules(6)
    # test_skope_rules(clf, 6, [58, 0, 20, 2, 14])
    #
    # clf = load_skope_rules(7)
    # test_skope_rules(clf, 7, [41, 5, 38, 57, 61, 28, 58, 17, 20, 14])
    #
    # clf = load_skope_rules(8)
    # test_skope_rules(clf, 8, [7, 0, 14, 57, 15, 25, 61, 4])
    #
    # clf = load_skope_rules(9)
    # test_skope_rules(clf, 9, [28, 29, 58, 0, 36, 16, 35, 61])
    #
    # clf = load_skope_rules(10)
    # test_skope_rules(clf, 10, [9, 15, 65, 18, 41, 33, 60, 2, 40, 56, 35, 14, 11, 51, 58, 31, 69, 22, 25, 0, 39, 61, 29])
    #
    # clf = load_skope_rules(11)
    # test_skope_rules(clf, 11, [36, 16, 61])
    #
    # clf = load_skope_rules(12)
    # test_skope_rules(clf, 12, [63, 3, 4, 68, 56, 66, 67, 57, 64, 50, 62, 47, 69, 34, 55, 65, 45, 37, 5, 8, 25, 18, 23, 60])
    #
    # clf = load_skope_rules(13)
    # test_skope_rules(clf, 13, [51, 21, 49, 58, 61, 0, 18, 2, 35, 14])
    #
    # clf = load_skope_rules(14)
    # test_skope_rules(clf, 14, [47, 8, 36, 33, 0, 34])
    #
    # denormalize_rules()
    #
    # test_multiclass_classification(0, True)