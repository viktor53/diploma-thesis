import numpy as np
from constants import PATH_TO_PRPD_DATA, NORMALIZED_DATA_DIR
from os import sep
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import IncrementalPCA


def run_pca(path_to_data: str):
    X_memmap = np.memmap(path_to_data + sep + "X.npy", dtype=np.float, mode="r", shape=(12986236, 78))

    logging.info("Running PCA fit.")

    transformer = IncrementalPCA(n_components=3, batch_size=100000)
    transformer.fit(X_memmap)

    logging.info("Running PCA transform.")

    X_transformed = transformer.transform(X_memmap)

    logging.info("Plotting results.")

    fig = plt.figure(1, figsize=(10, 9))
    plt.clf()
    ax = Axes3D(fig, rect=(0, 0, .95, 1), elev=48, azim=134)
    plt.cla()

    Y_memmap = np.memmap(path_to_data + sep + "Y.npy", dtype=np.float, mode="r", shape=12986236)

    ax.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2], c=Y_memmap, cmap=plt.cm.nipy_spectral,
               edgecolor='k')


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s", level=logging.INFO)

    run_pca(PATH_TO_NORM_DATA)
