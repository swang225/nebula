import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def to_pca(df, n_components):
    data_matrix = list(df["image"].apply(lambda x: x.flatten()).values)

    pca = PCA().fit(data_matrix)
    eigen_imgs = pca.components_[:n_components]

    weights = eigen_imgs @ (data_matrix - pca.mean_).T
    weights = weights.transpose()

    return weights, eigen_imgs, pca