from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_validate
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as snb
from collections import Counter
from constants import PATH_TO_NORM_DATA, CLASSES, FULL_TRAIN_SIZE, TRAIN_SIZE, VALIDATION_SIZE, SMALL_TRAIN_SIZE, SMALLEST_TRAIN_SIZE, TEST_SIZE
from os import sep
import logging
import numpy as np
from typing import List, Tuple, Callable, Dict
from timeit import default_timer as timer
from joblib import dump, load


BEST_WEIGHTS_FOR_LOG_REG = [
    [2.95339791, 0.60189949 * 3],
    [0.50897327 * 3.3113, 28.36051463],
    [5.00006882e-01 * 3, 3.63250594e+04],
    [5.00018626e-01 * 1.5, 1.34224380e+04],
    [0.52206274 * 7., 11.83132367],
    [5.00053187e-01, 4.70089005e+03],
    [0.51840077 * 1.7625, 14.08639111],
    [0.5012817 * 2.6647, 195.55334488],
    [0.51464435 * 3.65, 17.5714293],
    [0.50434616 * 6.5, 58.02206622],
    [5.00338617e-01 * 1.65, 7.38797255e+02],
    [0.50602754 * 4.6, 41.97630265],
    [0.50503736 * 1.25, 50.12915694],
    [5.00002503e-01 * 1.75, 9.98939135e+04],
    [0.50584556 * 7.78, 43.26753155]
]


def load_header() -> List[str]:
    with open(PATH_TO_NORM_DATA + sep + "train_full.csv", "r") as f:
        return f.readline()[:-1].split(",")[:-1]


def load_full_train_npy() -> Tuple[np.memmap, np.memmap]:
    logging.info("Loading full train dataset.")

    logging.info("Loading X with shape ({}, 70).".format(FULL_TRAIN_SIZE))
    X_memmap = np.memmap(PATH_TO_NORM_DATA + sep + "full_train" + sep + "X_full_train.npy", dtype=np.float32, mode="c",
                         shape=(FULL_TRAIN_SIZE, 70))

    logging.info("Loading Y with shape ({},).".format(FULL_TRAIN_SIZE))
    Y_memmap = np.memmap(PATH_TO_NORM_DATA + sep + "full_train" + sep + "Y_full_train.npy", dtype=np.int, mode="c",
                         shape=FULL_TRAIN_SIZE)

    logging.info("Loading is completed.")

    return X_memmap, Y_memmap


def load_full_train_npy_cls(cls: int) -> Tuple[np.memmap, np.memmap]:
    logging.info("Loading full train dataset.")

    logging.info("Loading X with shape ({}, 70).".format(FULL_TRAIN_SIZE))
    X_memmap = np.memmap(PATH_TO_NORM_DATA + sep + "full_train" + sep + "X_full_train.npy", dtype=np.float32, mode="c",
                         shape=(FULL_TRAIN_SIZE, 70))

    logging.info("Loading Y with shape ({},).".format(FULL_TRAIN_SIZE))
    Y_memmap = np.memmap(PATH_TO_NORM_DATA + sep + "full_train" + sep + "Y_full_train_cls_{}.npy".format(cls), dtype=np.int, mode="c",
                         shape=FULL_TRAIN_SIZE)

    logging.info("Loading is completed.")

    return X_memmap, Y_memmap


def load_test_npy() -> Tuple[np.memmap, np.memmap]:
    logging.info("Loading test dataset.")

    logging.info("Loading X with shape ({}, 70).".format(TEST_SIZE))
    X_memmap = np.memmap(PATH_TO_NORM_DATA + sep + "test" + sep + "X_test.npy", dtype=np.float32, mode="c",
                         shape=(TEST_SIZE, 70))

    logging.info("Loading Y with shape ({},).".format(TEST_SIZE))
    Y_memmap = np.memmap(PATH_TO_NORM_DATA + sep + "test" + sep + "Y_test.npy", dtype=np.int, mode="c",
                         shape=TEST_SIZE)

    logging.info("Loading is completed.")

    return X_memmap, Y_memmap


def load_test_npy_cls(cls: int) -> Tuple[np.memmap, np.memmap]:
    logging.info("Loading test dataset.")

    logging.info("Loading X with shape ({}, 70).".format(TEST_SIZE))
    X_memmap = np.memmap(PATH_TO_NORM_DATA + sep + "test" + sep + "X_test.npy", dtype=np.float32, mode="c",
                         shape=(TEST_SIZE, 70))

    logging.info("Loading Y with shape ({},).".format(TEST_SIZE))
    Y_memmap = np.memmap(PATH_TO_NORM_DATA + sep + "test" + sep + "Y_test_cls_{}.npy".format(cls), dtype=np.int, mode="c",
                         shape=TEST_SIZE)

    logging.info("Loading is completed.")

    return X_memmap, Y_memmap


def load_train_npy() -> Tuple[np.memmap, np.memmap]:
    logging.info("Loading train dataset.")

    logging.info("Loading X with shape ({}, 70).".format(TRAIN_SIZE))
    X_memmap = np.memmap(PATH_TO_NORM_DATA + sep + "train" + sep + "X_train.npy", dtype=np.float32, mode="c",
                         shape=(TRAIN_SIZE, 70))

    logging.info("Loading Y with shape ({},).".format(TRAIN_SIZE))
    Y_memmap = np.memmap(PATH_TO_NORM_DATA + sep + "train" + sep + "Y_train.npy", dtype=np.int, mode="c",
                         shape=TRAIN_SIZE)

    logging.info("Loading is completed.")

    return X_memmap, Y_memmap


def load_train_npy_cls(cls: int) -> Tuple[np.memmap, np.memmap]:
    logging.info("Loading train dataset.")

    logging.info("Loading X with shape ({}, 70).".format(TRAIN_SIZE))
    X_memmap = np.memmap(PATH_TO_NORM_DATA + sep + "train" + sep + "X_train.npy", dtype=np.float32,
                         mode="c", shape=(TRAIN_SIZE, 70))

    logging.info("Loading Y with shape ({},).".format(TRAIN_SIZE))
    Y_memmap = np.memmap(PATH_TO_NORM_DATA + sep + "train" + sep + "Y_train_cls_{}.npy".format(cls),
                         dtype=np.int, mode="c", shape=TRAIN_SIZE)

    logging.info("Loading is completed.")

    return X_memmap, Y_memmap


def load_small_train_npy() -> Tuple[np.memmap, np.memmap]:
    logging.info("Loading small train dataset.")

    logging.info("Loading X with shape ({}, 70).".format(SMALL_TRAIN_SIZE))
    X_memmap = np.memmap(PATH_TO_NORM_DATA + sep + "small_train" + sep + "X_small_train.npy", dtype=np.float32, mode="c",
                         shape=(SMALL_TRAIN_SIZE, 70))

    logging.info("Loading Y with shape ({},).".format(SMALL_TRAIN_SIZE))
    Y_memmap = np.memmap(PATH_TO_NORM_DATA + sep + "small_train" + sep + "Y_small_train.npy", dtype=np.int, mode="c",
                         shape=SMALL_TRAIN_SIZE)

    logging.info("Loading is completed.")

    return X_memmap, Y_memmap


def load_small_train_npy_cls(cls: int) -> Tuple[np.memmap, np.memmap]:
    logging.info("Loading small train dataset.")

    logging.info("Loading X with shape ({}, 70).".format(SMALL_TRAIN_SIZE))
    X_memmap = np.memmap(PATH_TO_NORM_DATA + sep + "small_train" + sep + "X_small_train.npy", dtype=np.float32,
                         mode="c", shape=(SMALL_TRAIN_SIZE, 70))

    logging.info("Loading Y with shape ({},).".format(SMALL_TRAIN_SIZE))
    Y_memmap = np.memmap(PATH_TO_NORM_DATA + sep + "small_train" + sep + "Y_small_train_cls_{}.npy".format(cls),
                         dtype=np.int, mode="c", shape=SMALL_TRAIN_SIZE)

    logging.info("Loading is completed.")

    return X_memmap, Y_memmap


def load_smallest_train_npy() -> Tuple[np.memmap, np.memmap]:
    logging.info("Loading smallest train dataset.")

    logging.info("Loading X with shape ({}, 70).".format(SMALLEST_TRAIN_SIZE))
    X_memmap = np.memmap(PATH_TO_NORM_DATA + sep + "small_train" + sep + "X_smaller_small_train.npy", dtype=np.float32, mode="c",
                         shape=(SMALLEST_TRAIN_SIZE, 70))

    logging.info("Loading Y with shape ({},).".format(SMALLEST_TRAIN_SIZE))
    Y_memmap = np.memmap(PATH_TO_NORM_DATA + sep + "small_train" + sep + "Y_smaller_small_train.npy", dtype=np.int, mode="c",
                         shape=SMALLEST_TRAIN_SIZE)

    logging.info("Loading is completed.")

    return X_memmap, Y_memmap


def load_smallest_train_npy_cls(cls: int) -> Tuple[np.memmap, np.memmap]:
    logging.info("Loading smallest train dataset.")

    logging.info("Loading X with shape ({}, 70).".format(SMALLEST_TRAIN_SIZE))
    X_memmap = np.memmap(PATH_TO_NORM_DATA + sep + "small_train" + sep + "X_smaller_small_train.npy", dtype=np.float32,
                         mode="c", shape=(SMALLEST_TRAIN_SIZE, 70))

    logging.info("Loading Y with shape ({},).".format(SMALLEST_TRAIN_SIZE))
    Y_memmap = np.memmap(PATH_TO_NORM_DATA + sep + "small_train" + sep + "Y_smaller_small_train_cls_{}.npy".format(cls),
                         dtype=np.int, mode="c", shape=SMALLEST_TRAIN_SIZE)

    logging.info("Loading is completed.")

    return X_memmap, Y_memmap


def load_validation_npy() -> Tuple[np.memmap, np.memmap]:
    logging.info("Loading validation dataset.")

    logging.info("Loading X with shape ({}, 70).".format(VALIDATION_SIZE))
    X_memmap = np.memmap(PATH_TO_NORM_DATA + sep + "validation" + sep + "X_validation.npy", dtype=np.float32, mode="c",
                         shape=(VALIDATION_SIZE, 70))

    logging.info("Loading Y with shape ({},).".format(VALIDATION_SIZE))
    Y_memmap = np.memmap(PATH_TO_NORM_DATA + sep + "validation" + sep + "Y_validation.npy", dtype=np.int, mode="c",
                         shape=VALIDATION_SIZE)

    logging.info("Loading is completed.")

    return X_memmap, Y_memmap


def load_validation_npy_cls(cls: int) -> Tuple[np.memmap, np.memmap]:
    logging.info("Loading validation dataset.")

    logging.info("Loading X with shape ({}, 70).".format(VALIDATION_SIZE))
    X_memmap = np.memmap(PATH_TO_NORM_DATA + sep + "validation" + sep + "X_validation.npy", dtype=np.float32,
                         mode="c", shape=(VALIDATION_SIZE, 70))

    logging.info("Loading Y with shape ({},).".format(VALIDATION_SIZE))
    Y_memmap = np.memmap(PATH_TO_NORM_DATA + sep + "validation" + sep + "Y_validation_cls_{}.npy".format(cls),
                         dtype=np.int, mode="c", shape=VALIDATION_SIZE)

    logging.info("Loading is completed.")

    return X_memmap, Y_memmap


def store_array(array: np.ndarray, name: str):
    memmap_array = np.memmap(name, dtype=np.float32, mode="write", shape=array.shape)
    memmap_array[:] = array[:]
    del memmap_array


def load_array(name: str, shape: Tuple[int, int]) -> np.ndarray:
    memmap_array = np.memmap(name, dtype=np.float32, mode="c", shape=shape)

    return np.array(memmap_array)


def train_random_forest(path_to_stored_model: str, load_train: Callable[[], Tuple[np.memmap, np.memmap]],
                        class_weight=None) -> RandomForestClassifier:
    rf = RandomForestClassifier(class_weight=class_weight, n_estimators=200, random_state=42, n_jobs=3, verbose=1)

    X_train, Y_train = load_train()

    logging.info("Running training of Random Forest.")

    start = timer()
    rf.fit(X_train, Y_train)
    end = timer()

    logging.info("Training is completed. (Took {:.2f} minutes)".format((end - start) / 60.))

    logging.info("Storing Random Forest model.")

    dump(rf, path_to_stored_model)

    return rf


def load_random_forest(path_to_stored_model: str) -> RandomForestClassifier:
    logging.info("Loading Random Forest model.")

    start = timer()
    rf = load(path_to_stored_model)
    end = timer()

    logging.info("Loading is completed. (Took {:.2f} minutes)".format((end - start) / 60.))

    return rf


def train_ada_boost(path_to_stored_model: str, load_train: Callable[[], Tuple[np.memmap, np.memmap]]) -> AdaBoostClassifier:
    ab = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2, random_state=42), n_estimators=400,
                            random_state=42)

    X_train, Y_train = load_train()

    logging.info("Running training of Ada Boost with Decision Tree.")

    start = timer()
    ab.fit(X_train, Y_train)
    end = timer()

    logging.info("Training is completed. (Took {:.2f} minutes)".format((end - start) / 60.))

    logging.info("Storing Ada Boost with Decision Tree model.")

    dump(ab, path_to_stored_model)

    return ab


def load_ada_boost(path_to_stored_model: str) -> AdaBoostClassifier:
    logging.info("Loading Ada Boost with Decision Tree model.")

    start = timer()
    ab = load(path_to_stored_model)
    end = timer()

    logging.info("Loading is completed. (Took {:.2f} minutes)".format((end - start) / 60.))

    return ab


def train_svc(path_to_stored_model: str, load_train: Callable[[], Tuple[np.memmap, np.memmap]],
              class_weight=None) -> SVC:
    svc = SVC(kernel='poly', cache_size=256, class_weight=class_weight, random_state=42, verbose=2)

    X_train, Y_train = load_train()

    logging.info("Running training of SVC.")

    start = timer()
    svc.fit(X_train, Y_train)
    end = timer()

    logging.info("Training is completed. (Took {:.2f} minutes)".format((end - start) / 60.))

    logging.info("Storing SVC model.")

    dump(svc, path_to_stored_model)

    return svc


def load_svc(path_to_stored_model: str) -> SVC:
    logging.info("Loading SVC model.")

    start = timer()
    svc = load(path_to_stored_model)
    end = timer()

    logging.info("Loading is completed. (Took {:.2f} minutes)".format((end - start) / 60.))

    return svc


def train_logistic_regression(path_to_stored_model: str, load_train: Callable[[], Tuple[np.memmap, np.memmap]],
                              class_weight=None) -> LogisticRegression:
    lr = LogisticRegression(class_weight=class_weight, C=0.1, penalty='l1', solver='saga', random_state=42, n_jobs=4,
                            verbose=1)

    X_train, Y_train = load_train()

    logging.info("Running training of Logistic Regression.")

    start = timer()
    lr.fit(X_train, Y_train)
    end = timer()

    logging.info("Training is completed. (Took {:.2f} minutes)".format((end - start) / 60.))

    logging.info("Storing Logistic Regression model.")

    dump(lr, path_to_stored_model)

    logging.info("Storing is completed.")

    return lr


def load_logistic_regression(path_to_stored_model: str) -> LogisticRegression:
    logging.info("Loading Logistic Regression model.")

    start = timer()
    lr = load(path_to_stored_model)
    end = timer()

    logging.info("Loading is completed. (Took {:.2f} minutes)".format((end - start) / 60.))

    return lr


def train_decision_tree(path_to_stored_model: str, load_train: Callable[[], Tuple[np.memmap, np.memmap]],
                        class_weight=None) -> DecisionTreeClassifier:
    dt = DecisionTreeClassifier(criterion="gini", class_weight=class_weight, random_state=42)

    X_train, Y_train = load_train()

    logging.info("Running training of Decision Tree.")

    start = timer()
    dt.fit(X_train, Y_train)
    end = timer()

    logging.info("Training is completed. (Took {:.2f} minutes)".format((end - start) / 60.))

    logging.info("Storing Decision Tree model.")

    dump(dt, path_to_stored_model)

    logging.info("Storing is completed.")

    return dt


def load_decision_tree(path_to_stored_model: str) -> DecisionTreeClassifier:
    logging.info("Loading Decision Tree model.")

    start = timer()
    dt = load(path_to_stored_model)
    end = timer()

    logging.info("Loading is completed. (Took {:.2f} minutes)".format((end - start) / 60.))

    return dt


def compute_accuracy(model, load_data: Callable[[], Tuple[np.memmap, np.memmap]], number_of_classes: int = 1) -> Tuple[float, float, float, np.ndarray]:
    X_data, Y_true = load_data()

    logging.info("Computing accuracy.")

    start = timer()
    Y_predict = model.predict(X_data)

    acc = accuracy_score(Y_true, Y_predict)

    if number_of_classes == 1:
        f1 = f1_score(Y_true, Y_predict)
        rec = recall_score(Y_true, Y_predict)
        prec = precision_score(Y_true, Y_predict)
    else:
        f1 = f1_score(Y_true, Y_predict, average='micro')
        rec = recall_score(Y_true, Y_predict, average='micro')
        prec = precision_score(Y_true, Y_predict, average='micro')

    end = timer()

    logging.info("Accuracy: {:.2f}%, F1 Score: {:.2f}, Recall: {:.2f} Precision: {:.2f} (Took {:.2f} minutes)".format(acc * 100, f1, rec, prec, (end - start) / 60.))

    return acc, f1, rec, Y_predict


def plot_confusion_matrix(model, data_loader: Callable[[], Tuple[np.memmap, np.memmap]]):
    x, y = data_loader()

    acc, f1, rec, y_pred = compute_accuracy(model, data_loader, 14)

    matrix = confusion_matrix(y, y_pred, normalize='true')

    logging.info("Plotting confusion matrix.")

    fig, ax = plt.subplots(figsize=(14, 7))
    snb.heatmap(matrix, annot=True, cbar=False, cmap=plt.cm.Reds, linewidths=0.5, ax=ax)

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")

    plt.gca().set_position((.2, .15, 0.75, 0.8))
    plt.figtext(.01, .4, "\n".join(["{} - {}".format(i, cls) for i, cls in enumerate(CLASSES)]))
    plt.figtext(.46, .025, "Accuracy - {:.2f}, F1 Score - {:.2f}, Recall - {:.2f}".format(acc * 100, f1, rec))

    plt.savefig("confusion_matrix.png", dpi=300)

    logging.info("Plotting is completed.")


def plot_binary_confusion_matrix(y_pred: np.ndarray, data_loader: Callable[[], Tuple[np.memmap, np.memmap]], cls: int,
                                 acc: float, f1: float, recall: float):
    _, y = data_loader()

    matrix = confusion_matrix(y, y_pred, normalize='true')

    logging.info("Plotting confusion matrix.")

    fig, ax = plt.subplots(figsize=(10, 11))
    snb.heatmap(matrix, annot=True, cbar=False, cmap=plt.cm.Reds, linewidths=0.5, ax=ax, annot_kws={'size': 24})

    plt.ylabel("True Label", fontsize=20)
    plt.xlabel("Predicted Label", fontsize=20)
    plt.title("Confusion Matrix for class {}".format(CLASSES[cls]), fontsize=26)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.gca().set_position((.12, .1, 0.8, 0.8))
    plt.figtext(.17, .015, "Accuracy - {:.2f}, F1 Score - {:.2f}, Recall - {:.2f}".format(acc * 100, f1, recall),
                fontsize=22)

    plt.savefig("confusion_matrix_{}.png".format(CLASSES[cls].replace(" ", "_").lower()), dpi=300)

    logging.info("Plotting is completed.")


def plot_random_forest_feature_imp(model_with_feature_imp):
    feature_importances = model_with_feature_imp.feature_importances_
    sorted_idx = feature_importances.argsort()

    header = np.array(load_header())

    y_ticks = np.arange(0, 2*len(header), 2)
    fig, ax = plt.subplots(figsize=(7, 10))
    ax.barh(y_ticks, feature_importances[sorted_idx])
    ax.set_yticklabels(header[sorted_idx])
    ax.set_yticks(y_ticks)
    ax.set_title("Random Forest Feature Importances")
    fig.tight_layout()

    plt.savefig("feature_importance.png", dpi=300)


def plot_feature_importance_from_model(model_with_coef):
    coef = model_with_coef.coef_

    header = np.array(load_header())

    i = 0
    for cls_feature_imp in coef:
        imps = np.abs(cls_feature_imp)
        sorted_idx = np.argsort(imps)

        y_ticks = np.arange(0, 2 * len(header), 2)
        plt.clf()
        fig, ax = plt.subplots(figsize=(7, 10))
        ax.barh(y_ticks, imps[sorted_idx])
        ax.set_yticklabels(header[sorted_idx])
        ax.set_yticks(y_ticks)
        ax.set_title("{} Feature Importances".format(CLASSES[i]))
        fig.tight_layout()

        plt.savefig("feature_importance_class_{}.png".format(i), dpi=300)

        i += 1


def plot_feature_importance_from_coef(coef: np.ndarray, cls: int):
    header = np.array(load_header())

    imps = np.abs(coef)
    sorted_idx = np.argsort(imps)

    y_ticks = np.arange(0, 2 * len(header), 2)
    plt.clf()
    fig, ax = plt.subplots(figsize=(7, 10))
    ax.barh(y_ticks, imps[sorted_idx])
    ax.set_yticklabels(header[sorted_idx])
    ax.set_yticks(y_ticks)
    ax.set_title("{} Feature Importances".format(CLASSES[cls]))
    fig.tight_layout()

    plt.savefig("feature_importance_class_{}.png".format(cls), dpi=300)


def plot_permutation_importance(model_with_feature_imp, iter: int = 0):
    X_val, Y_val = load_validation_npy()

    logging.info("Computing permutation importance.")

    start = timer()
    result = permutation_importance(model_with_feature_imp, X_val, Y_val, n_repeats=3,
                                    random_state=42, n_jobs=1)
    end = timer()

    logging.info("Computing is completed. (Took {:.2f} minutes)".format((end - start) / 60))

    sorted_idx = result.importances_mean.argsort()

    if iter > 0:
        prev = load_array("perm_imp_data_{}.npy".format(iter - 1), shape=(70, 3 * iter))
        next = np.concatenate((prev, result.importances), axis=1)
    else:
        next = result

    sorted_idx = np.argsort(np.mean(next, 1))

    header = np.array(load_header())

    logging.info("Plotting result.")

    fig, ax = plt.subplots(figsize=(7, 10))
    ax.boxplot(next[sorted_idx].T,
               vert=False, labels=header[sorted_idx])
    ax.set_title("Permutation Importances")
    fig.tight_layout()

    plt.savefig("perm_feature_imp.png", dpi=300)

    store_array(next, "perm_imp_data_{}.npy".format(iter))


def get_class_weights(data_loader: Callable[[], Tuple[np.memmap, np.memmap]], n_classes=15) -> np.ndarray:
    _, y = data_loader()

    logging.info("Computing classes weights.")

    cntr = Counter()
    for cls in y:
        cntr.update([cls])

    n_samples = sum(cntr.values())

    result = np.zeros((n_classes,))

    for i in range(n_classes):
        result[i] = n_samples / (n_classes * cntr[i])

    logging.info("Classes weights are computed. {}".format(result))

    return result


def train_logistic_regression_for_each_class():
    path_to_model = "../logistic_regression/logistic_regression_cls_{}.joblib"
    for cls in range(15):
        class_weights = get_class_weights(lambda: load_train_npy_cls(cls), n_classes=2)
        class_weights_dict = dict(zip(list(range(0, 2)), class_weights))
        lr = train_logistic_regression(path_to_model.format(cls), lambda: load_train_npy_cls(cls), class_weights_dict)
        acc, f1, rec, y_pred = compute_accuracy(lr, lambda: load_validation_npy_cls(cls))
        plot_binary_confusion_matrix(y_pred, lambda: load_train_npy_cls(cls), cls, acc, f1, rec)


def log_reg_feature_importance(path_to_model: str):
    path_to_model_for_cls = path_to_model + sep + "logistic_regression_cls_{}.joblib"

    for cls in range(15):
        lr = load_logistic_regression(path_to_model_for_cls.format(cls))
        plot_feature_importance_from_coef(lr.coef_[0], cls)


def get_best_features_log_reg(path_to_model: str, number_of_features: int) -> Tuple[np.ndarray, np.ndarray]:
    path_to_model_for_cls = path_to_model + sep + "logistic_regression_cls_{}.joblib"

    logging.info("Extracting most important features from logistic regression.")

    result = set()

    for cls in range(15):
        lr = load_logistic_regression(path_to_model_for_cls.format(cls))
        imps = np.abs(lr.coef_[0])
        sorted_idx = np.argsort(imps)

        result = result.union(sorted_idx[-number_of_features:])

    header = np.array(load_header())

    logging.info("Extracted features from logistic regression: {}".format(len(result)))

    indexes = np.array(list(result))
    columns_names = header[indexes]

    return indexes, columns_names


def get_best_features_decision_tree(path_to_model: str, number_of_features: int) -> Tuple[np.ndarray, np.ndarray]:
    logging.info("Extracting most important features from decision tree.")

    dt = load_decision_tree(path_to_model)

    features = dt.feature_importances_.argsort()
    header = np.array(load_header())
    header = header[features]

    return features[-number_of_features:], header[-number_of_features:]


def logistic_regression_predict(path_to_model: str, x: np.ndarray) -> np.ndarray:
    path_to_model = path_to_model + sep + "logistic_regression_cls_{}.joblib"

    lr_models = []

    for cls in range(15):
        lr_models.append(load_logistic_regression(path_to_model.format(cls)))

    logging.info("Predicting.")

    start = timer()

    models_predict = []
    for lr in lr_models:
        models_predict.append(lr.predict_proba(x)[:, 1])

    end = timer()

    logging.info("Predicting is completed. (Took {:.2f} minutes)".format((end - start) / 60.))

    return np.argmax(models_predict, axis=0)


def logistic_regression_accuracy(path_to_model: str, data_loader: Callable[[], Tuple[np.memmap, np.memmap]]) -> Tuple[float, float, float]:
    x, y = data_loader()

    predict = logistic_regression_predict(path_to_model, x)

    acc = accuracy_score(y, predict)

    f1 = f1_score(y, predict, average='micro')

    rec = recall_score(y, predict, average='micro')

    logging.info("Accuracy: {:.2f}, F1 Score: {:.2f}, Recall: {:.2f}".format(acc * 100, f1, rec))

    return acc, f1, rec


def run_anova(data_loader: Callable[[], Tuple[np.memmap, np.memmap]]) -> np.ndarray:
    X, Y = data_loader()

    anova = SelectKBest(score_func=f_classif, k=50)
    anova.fit(X, Y)

    mean = np.mean(anova.scores_)
    std = np.std(anova.scores_)

    return (np.array(anova.scores_) - mean) / std


def plot_anova_feature_importance():
    for cls in range(15):
        logging.info("Running ANOVA for class: {}".format(cls))

        start = timer()
        imps = run_anova(lambda: load_train_npy_cls(cls))
        plot_feature_importance_from_coef(imps, cls)
        end = timer()

        logging.info("ANOVA is completed. (Took {:.2f} minutes)".format((end - start) / 60.))


def run_cross_validation(model, data_loader: Callable[[], Tuple[np.memmap, np.memmap]], cv: int = 5) -> Dict:
    X, Y = data_loader()

    logging.info("Running cross validation.")

    start = timer()
    score = cross_validate(model, X, Y, cv=cv, scoring=['accuracy', 'f1_micro', 'recall_micro'], verbose=2)
    end = timer()

    logging.info("Cross validation is completed. (Took {:.2f} minutes)".format((end - start) / 60.))

    logging.info("Accuracy: {}".format(score['test_accuracy']))
    logging.info("F1 Score: {}".format(score['test_f1_micro']))
    logging.info("Recall: {}".format(score['test_recall_micro']))

    return score


def provide_only_best_features(data_loader: Callable[[], Tuple[np.memmap, np.memmap]],
                               feature_loader: Callable[[int], Tuple[np.ndarray, np.ndarray]],
                               number_of_features: int = 10) -> Tuple[np.memmap, np.memmap]:
    indxs, col_names = feature_loader(number_of_features)

    logging.info("Best features names: {}".format(col_names))
    logging.info("Best features indexes: {}".format(indxs))

    X, Y = data_loader()

    return X[:, indxs], Y


def provide_only_best_features_tree(data_loader: Callable[[], Tuple[np.memmap, np.memmap]], cls: int,
                                    number_of_features: int) -> Tuple[np.memmap, np.memmap]:
    path_to_model = "../decision_tree_cls_{}.joblib".format(cls)

    return provide_only_best_features(data_loader, lambda n: get_best_features_decision_tree(path_to_model, n),
                                      number_of_features)


def compute_ttest_for_models(result_1: Dict, result_2: Dict):
    var_1 = np.var(result_1['test_accuracy'])
    var_2 = np.var(result_2['test_accuracy'])
    var1_equal = var_1 == var_2

    logging.info("Variance_1: {:.4f}, Variance_2: {:.4f}".format(var_1, var_2))

    var_1 = np.var(result_1['test_f1_micro'])
    var_2 = np.var(result_2['test_f1_micro'])
    var2_equal = var_1 == var_2

    logging.info("Variance_1: {:.4f}, Variance_2: {:.4f}".format(var_1, var_2))

    var_1 = np.var(result_1['test_recall_micro'])
    var_2 = np.var(result_2['test_recall_micro'])
    var3_equal = var_1 == var_2

    logging.info("Variance_1: {:.4f}, Variance_2: {:.4f}".format(var_1, var_2))

    t, prob1 = ttest_ind(result_1['test_accuracy'], result_2['test_accuracy'],
                         equal_var=var1_equal)

    logging.info("ttest p-value for accuracy: {:.4f}".format(prob1))

    t, prob2 = ttest_ind(result_1['test_f1_micro'], result_2['test_f1_micro'],
                         equal_var=var2_equal)

    logging.info("ttest p-value for f1: {:.4f}".format(prob2))

    t, prob3 = ttest_ind(result_1['test_recall_micro'], result_2['test_recall_micro'],
                         equal_var=var3_equal)

    logging.info("ttest p-value for recall: {:.4f}".format(prob3))

    return prob1, prob2, prob3


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s", level=logging.INFO)

    # train_random_forest("../random_forest.joblib", load_train_npy)
    # rf = load_random_forest("../random_forest.joblib")
    # plot_random_forest_feature_imp(rf)
    # plot_permutation_importance(rf, 1)
    # plot_confusion_matrix(rf, load_validation_npy)

    # train_logistic_regression_for_each_class()
    # logistic_regression_accuracy("../logistic_regression", load_validation_npy)

    # cls = 0
    # weights = [1., 1.]
    # class_weights = dict(zip(list(range(0, 2)), weights))
    # lr = train_logistic_regression("../best_logistic_regression/logistic_regression_cls_{}.joblib".format(cls),
    #                                lambda: load_train_npy_cls(cls), class_weights)
    # for cls in range(15):
    #     lr = load_logistic_regression("../log_reg/logistic_regression_cls_{}.joblib".format(cls))
    #     acc, f1, rec, predicted = compute_accuracy(lr, lambda: load_test_npy_cls(cls))
    #     plot_binary_confusion_matrix(predicted, lambda: load_test_npy_cls(cls), cls, acc, f1, rec)

    # log_reg_feature_importance("../best_logistic_regression")
    # get_best_features_log_reg("../best_logistic_regression", 15)

    # plot_feature_importance_from_coef(run_anova(load_train_npy), 0)
    # plot_anova_feature_importance()

    # run_cross_validation_random_forest(load_full_train_npy)
    # run_cross_validation_random_forest(lambda: provide_only_best_features(load_full_train_npy))
    # compute_ttest()

    # ab = train_ada_boost("../ada_boost_2.joblib", load_small_train_npy)
    # plot_confusion_matrix(ab, load_validation_npy)
    # plot_random_forest_feature_imp(ab)

    # svc = train_svc("../svc.joblib", load_smallest_train_npy)
    # plot_confusion_matrix(svc, load_validation_npy)

    # for cls in [3, 9, 12, 13]:
    #     svc = train_svc("../svc_cls_{}.joblib".format(cls), lambda: load_smallest_train_npy_cls(cls), 'balanced')
    #     acc, f1, rec = compute_accuracy(svc, lambda: load_validation_npy_cls(cls))
    #     plot_binary_confusion_matrix(svc, lambda: load_validation_npy_cls(cls), cls, acc, f1, rec)

    # cls = 3
    # weights = get_class_weights(lambda: load_smallest_train_npy_cls(cls), n_classes=2)
    # weights = [5.00009062e-01, 2.75893000e+04]
    # class_weights = dict(zip(list(range(0, 2)), weights))
    # svc = train_svc("../svc_cls_{}.joblib".format(cls), lambda: load_smallest_train_npy_cls(cls), class_weight=class_weights)
    # acc, f1, rec = compute_accuracy(svc, lambda: load_validation_npy_cls(cls))
    # plot_binary_confusion_matrix(svc, lambda: load_validation_npy_cls(cls), cls, acc, f1, rec)

    # cls = 5
    # # weights = get_class_weights(lambda: load_train_npy_cls(cls), n_classes=2)
    # # weights = [5.00002503e-01, 9.98939135e+04 * 0.5]
    # # class_weights = dict(zip(list(range(0, 2)), weights))
    # class_weights = None
    # is_cross_val = False
    # if not is_cross_val:
    #     # model = train_random_forest("../random_forest_cls_{}.joblib".format(cls), lambda: load_train_npy_cls(cls), class_weights)
    #     model = train_decision_tree("../decision_tree_cls_{}.joblib".format(cls), lambda: load_train_npy_cls(cls), class_weights)
    #     # model = load_decision_tree("../decision_tree_cls_{}.joblib".format(cls))
    #     acc, f1, rec, y_pred = compute_accuracy(model, lambda: load_validation_npy_cls(cls))
    #     plot_binary_confusion_matrix(y_pred, lambda: load_validation_npy_cls(cls), cls, acc, f1, rec)
    #     plot_feature_importance_from_coef(model.feature_importances_, cls)
    # else:
    #     result_1 = run_cross_validation(DecisionTreeClassifier(random_state=42, class_weight=class_weights), lambda: load_full_train_npy_cls(cls), cv=5)
    #     result_2 = run_cross_validation(DecisionTreeClassifier(random_state=42, class_weight=class_weights),
    #                                     lambda: provide_only_best_features_tree(lambda: load_full_train_npy_cls(cls), cls, 25), cv=5)
    #     compute_ttest_for_models(result_1, result_2)

    # cls = 0
    # features = 4
    # # class_weights = dict(zip(list(range(0, 2)), [5.00002503e-01, 9.98939135e+04 * 0.5]))
    # class_weights = None
    # tree = train_decision_tree("../final_decision_tree/decision_tree_cls_{}.joblib".format(cls),
    #                            lambda: provide_only_best_features_tree(lambda: load_full_train_npy_cls(cls), cls, features),
    #                            class_weights)
    # acc, f1, rec, y_pred = compute_accuracy(tree, lambda: provide_only_best_features_tree(lambda: load_test_npy_cls(cls), cls, features))
    # plot_binary_confusion_matrix(y_pred, lambda: provide_only_best_features_tree(lambda: load_test_npy_cls(cls), cls, features),
    #                              cls, acc, f1, rec)

    # cls = 13
    # f_n = 12
    # path_to_model_for_cls = "../best_logistic_regression" + sep + "logistic_regression_cls_{}.joblib"
    # lr = load_logistic_regression(path_to_model_for_cls.format(cls))
    # imps = np.abs(lr.coef_[0])
    # sorted_idx = np.argsort(imps)
    # imps = imps[sorted_idx]
    # header = np.array(load_header())[sorted_idx]
    # print(np.flip(imps[-f_n:]))
    # print(", ".join(np.flip(header[-f_n:])))
    # f, h = get_best_features_decision_tree("../decision_tree_cls_{}.joblib".format(cls), f_n)
    # print(np.flip(f))
    # print(", ".join(np.flip(h)))

    # for cls in range(15):
    #     imps = run_anova(lambda: load_train_npy_cls(cls))
    #     header = np.array(load_header())
    #
    #     imps = np.abs(imps)
    #     sorted_idx = np.argsort(imps)
    #
    #     header = header[sorted_idx]
    #     imps = imps[sorted_idx]
    #
    #     print("**** class {} ****".format(cls))
    #     print(np.flip(imps))
    #     print(", ".join(np.flip(header)))
    #     print("******************")

    # for cls, features in zip(list(range(15)), [23, 2, 9, 20, 5, 1, 5, 10, 8, 8, 23, 3, 63, 10, 6]):
    #     lr = load_decision_tree("../final_decision_tree/decision_tree_cls_{}.joblib".format(cls))
    #     acc, f1, rec, predicted = compute_accuracy(lr, lambda: provide_only_best_features_tree(lambda: load_test_npy_cls(cls), cls, features))
    #     plot_binary_confusion_matrix(predicted, lambda: provide_only_best_features_tree(lambda: load_test_npy_cls(cls), cls, features),
    #                                  cls, acc, f1, rec)

    for cls in [3, 9, 12, 13]:
        svc = load_svc("../svc_cls_{}.joblib".format(cls))
        acc, f1, rec, predicted = compute_accuracy(svc, lambda: load_test_npy_cls(cls))
        plot_binary_confusion_matrix(predicted, lambda: load_test_npy_cls(cls), cls, acc, f1, rec)
