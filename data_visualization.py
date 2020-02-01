import numpy as np
from typing import Callable
from constants import PATH_TO_PRPD_DATA, PATH_TO_NORM_DATA
from os import sep
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import LocallyLinearEmbedding, MDS
from umap import UMAP


def run_pca_explained_variance(path_to_data: str, number_of_samples: int = 12986236, dimensions: int = 2):
    logging.info("Loading data.")

    X_memmap = np.memmap(path_to_data + sep + "X_train_full.npy", dtype=np.float32, mode="r", shape=(number_of_samples, 70))

    logging.info("Running PCA fit for {} components.".format(dimensions))

    transformer = IncrementalPCA(n_components=dimensions, batch_size=500000)
    transformer.fit(X_memmap)

    logging.info("PCA fit is completed.")

    return transformer.explained_variance_, transformer.explained_variance_ratio_


def run_pca(path_to_data: str, number_of_samples: int = 12986236, dimensions: int = 2,
            from_scratch: bool = True) -> str:
    if from_scratch:
        X_memmap = np.memmap(path_to_data + sep + "X_train_sample.npy", dtype=np.float32, mode="r", shape=(number_of_samples, 70))

        logging.info("Running PCA fit.")

        transformer = IncrementalPCA(n_components=dimensions, batch_size=500000)
        transformer.fit(X_memmap)

        logging.info("Running PCA transform.")

        X_transformed = transformer.transform(X_memmap)

        logging.info("Storing transformed X.")

        X_transformed_memmap = np.memmap(path_to_data + sep + "X_train_sample_pca_transformed_{}.npy".format(dimensions),
                                         dtype=np.float32, mode="write", shape=(number_of_samples, dimensions))

        X_transformed_memmap[:] = X_transformed[:]

        del X_transformed_memmap

    return "X_train_sample_pca_transformed_{}.npy".format(dimensions)


def run_locally_linear_embedding(path_to_data: str, number_of_samples: int = 12986236, dimensions: int = 2,
                                 from_scratch: bool = True) -> str:
    if from_scratch:
        X_memmap = np.memmap(path_to_data + sep + "X_train_sample.npy", dtype=np.float32, mode="r", shape=(number_of_samples, 70))

        logging.info("Running Locally Linear Embedding fit.")

        transformer = LocallyLinearEmbedding(n_neighbors=5, n_components=dimensions, n_jobs=4)
        transformer.fit(X_memmap)

        logging.info("Running Locally Linear Embedding transform.")

        X_transformed = transformer.transform(X_memmap)

        logging.info("Storing transformed X.")

        X_transformed_memmap = np.memmap(path_to_data + sep + "X_train_sample_lle_transformed_{}.npy".format(dimensions),
                                         dtype=np.float32, mode="write", shape=(number_of_samples, dimensions))

        X_transformed_memmap[:] = X_transformed[:]

        del X_transformed_memmap

    return "X_train_sample_lle_transformed_{}.npy".format(dimensions)


def run_multidimensional_scaling(path_to_data: str, number_of_samples: int = 12986236, dimensions: int = 2,
                                 from_scratch: bool = True) -> str:
    if from_scratch:
        X_memmap = np.memmap(path_to_data + sep + "X_train_sample.npy", dtype=np.float32, mode="r", shape=(number_of_samples, 70))

        logging.info("Running Multidimensional Scaling fit and transform.")

        transformer = MDS(n_components=dimensions, n_jobs=4)
        X_transformed = transformer.fit_transform(X_memmap)

        logging.info("Storing transformed X.")

        X_transformed_memmap = np.memmap(path_to_data + sep + "X_train_sample_mds_transformed_{}.npy".format(dimensions),
                                         dtype=np.float32, mode="write", shape=(number_of_samples, dimensions))

        X_transformed_memmap[:] = X_transformed[:]

        del X_transformed_memmap

    return "X_train_sample_mds_transformed_{}.npy".format(dimensions)


def run_uniform_manifold_approx(path_to_data: str, number_of_samples: int = 12986236, dimensions: int = 2,
                                from_scratch: bool = True) -> str:
    if from_scratch:
        X_memmap = np.memmap(path_to_data + sep + "X_train_sample.npy", dtype=np.float32, mode="r", shape=(number_of_samples, 70))

        logging.info("Running Uniform Manifold Approximation fit and transform.")

        transformer = UMAP(n_components=dimensions)
        X_transformed = transformer.fit_transform(X_memmap)

        logging.info("Storing transformed X.")

        X_transformed_memmap = np.memmap(path_to_data + sep + "X_train_sample_umap_transformed_{}.npy".format(dimensions),
                                         dtype=np.float32, mode="write", shape=(number_of_samples, dimensions))

        X_transformed_memmap[:] = X_transformed[:]

        del X_transformed_memmap

    return "X_train_sample_umap_transformed_{}.npy".format(dimensions)


def plot_3d_transformation(path_to_data: str, transformer: Callable[[str, int, int, bool], str], title: str,
                           number_of_samples: int = 12986236, from_scratch=True):
    data = transformer(path_to_data, number_of_samples, 3, from_scratch)

    logging.info("Loading transformed X.")

    X_transformed_memmap = np.memmap(path_to_data + sep + data, dtype=np.float32, mode="r",
                                     shape=(number_of_samples, 3))

    logging.info("Plotting results.")

    fig = plt.figure(1, figsize=(10, 9))
    plt.clf()
    ax = Axes3D(fig, rect=(0, 0, .95, 1))
    plt.cla()

    Y_memmap = np.memmap(path_to_data + sep + "Y_train_sample.npy", dtype=np.int, mode="r", shape=number_of_samples)

    ax.scatter(X_transformed_memmap[:, 0], X_transformed_memmap[:, 1], X_transformed_memmap[:, 2], c=Y_memmap,
               cmap=plt.cm.nipy_spectral, edgecolor='k')

    # ax.set_xlim(100, 160)
    # ax.set_ylim(-150, -100)
    # ax.set_zlim(50, 70)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.title(title)

    logging.info("Saving figure.")

    plt.savefig("{}.png".format(title.replace(" ", "_")), dpi=300)


def plot_transformation(path_to_data: str, transformer: Callable[[str, int, int, bool], str], title: str,
                        number_of_samples: int = 12986236, from_scratch=True):
    data = transformer(path_to_data, number_of_samples, 2, from_scratch)

    logging.info("Loading transformed X.")

    X_transformed_memmap = np.memmap(path_to_data + sep + data, dtype=np.float32, mode="r",
                                     shape=(number_of_samples, 2))

    logging.info("Plotting results.")

    fig, ax = plt.subplots(figsize=(10, 9))

    Y_memmap = np.memmap(path_to_data + sep + "Y_train_sample.npy", dtype=np.int, mode="r", shape=number_of_samples)

    ax.scatter(X_transformed_memmap[:, 0], X_transformed_memmap[:, 1], c=Y_memmap,
               cmap=plt.cm.nipy_spectral, edgecolor='k')

    ax.set_xlim(-5, 22)
    ax.set_ylim(-20, 17)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.title(title)

    logging.info("Saving figure.")

    plt.savefig("{}.png".format(title.replace(" ", "_")), dpi=300)


def plot_explained_variance(path_to_data: str, number_of_samples: int = 12986236):
    components = [i for i in range(1, 70, 2)]
    evs = []
    evrs = []

    logging.info("Running PCA.")

    for c in components:
        ev, evr = run_pca_explained_variance(path_to_data, number_of_samples, c)
        result_ev = sum(ev)
        result_evr = sum(evr)
        logging.info("Explained variance for {} components: {}".format(c, result_ev))
        logging.info("Explained variance ratio for {} components: {}".format(c, result_evr))
        evs.append(result_ev)
        evrs.append(result_evr)

    logging.info("Explained variances: {}".format(evs))
    logging.info("Explained variances ratio: {}".format(evrs))

    logging.info("Plotting result.")

    fig, ax = plt.subplots(figsize=(10, 9))

    ax.plot(components, evs)

    ax.set_xlabel("components")
    ax.set_ylabel("explained variance")

    plt.title("Explained Variance")

    plt.savefig("exp_var.png", dpi=300)

    plt.clf()

    fig, ax = plt.subplots(figsize=(10, 9))

    ax.plot(components, evrs)

    ax.set_xlabel("components")
    ax.set_ylabel("explained variance ratio")

    plt.title("Explained Variance Ratio")

    plt.savefig("exp_var_rat.png", dpi=300)


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s", level=logging.INFO)

    # plot_3d_transformation(PATH_TO_NORM_DATA, run_pca, "PCA 3", from_scratch=False)
    # plot_3d_transformation(PATH_TO_NORM_DATA, run_pca, "PCA 3", number_of_samples=2798546, from_scratch=False)

    # plot_transformation(PATH_TO_NORM_DATA, run_pca, "PCA", number_of_samples=2798546, from_scratch=False)
    # plot_transformation(PATH_TO_NORM_DATA, run_locally_linear_embedding, "LLE", number_of_samples=2798546,
    #                     from_scratch=True)
    # plot_transformation(PATH_TO_NORM_DATA, run_multidimensional_scaling, "MDS", number_of_samples=2798546,
    #                     from_scratch=True)
    # plot_transformation(PATH_TO_NORM_DATA, run_uniform_manifold_approx, "UMAP", number_of_samples=2798546,
    #                     from_scratch=True)
    # plot_explained_variance(PATH_TO_NORM_DATA)
