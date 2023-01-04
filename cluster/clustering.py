import logging
import random
import anndata
import numpy as np
import pandas as pd
import umap
import scanpy as sc
from matplotlib import pyplot as plt
from sklearn.metrics import *

logging.getLogger().setLevel(logging.INFO)


class Clustering:
    @staticmethod
    def data_read(data_path):
        """
        :param data_path: the matrix or the labels' path
        :return: the matrix or the label extracted by python from files
        """
        if data_path.endswith(".csv"):
            return pd.read_csv(data_path, sep=',', index_col=0)
        elif data_path.endswith(".tsv"):
            return pd.read_csv(data_path, sep="\t", index_col=0)
        elif data_path.endswith(".npy"):
            return np.load(data_path)
        elif data_path.endswith(".xls") or data_path.endswith(".xlsx"):
            return pd.read_excel(data_path, index_col=0)
        elif data_path.endswith(".h5ad"):
            return anndata.read_h5ad(data_path).X
        else:
            return "the data is not formatted!"

    @staticmethod
    def louvain_clustering(data):
        """
        :param data: dim == (n_cells*M)
        :return: the reduced data after umap, the prediction labels
        """

        reducer = umap.UMAP(random_state=123)
        reduced = reducer.fit_transform(data)

        adata = anndata.AnnData(X=np.random.random(size=data.shape), dtype=np.float64)
        adata.obsm['latent'] = data
        sc.pp.neighbors(adata, n_neighbors=30, use_rep='latent')

        sc.tl.louvain(adata)
        pred_labels = adata.obs['louvain'].tolist()
        pred_labels = np.array([int(i) for i in pred_labels], dtype=int)

        return reduced, pred_labels

    @staticmethod
    def index_compute(pred_labels, true_labels, batch=128):
        """
        :param pred_labels: 
        :param true_labels: 
        :return: the dataframe which includes the v_score, ari and the ami
        """
        label_true = np.array(true_labels)
        label_pred = np.array(pred_labels)

        label_true_split = np.array_split(label_true, len(label_true) // batch)
        label_pred_split = np.array_split(label_pred, len(label_pred) // batch)

        v_score, ari, ami = [], [], []

        # index impute
        for true, label in zip(label_true_split, label_pred_split):
            v_score.append(v_measure_score(true, label))
            ari.append(adjusted_rand_score(true, label))
            ami.append(adjusted_mutual_info_score(true, label))
        evaluations = pd.DataFrame({
            'v_score': v_score,
            'ari': ari,
            'ami': ami,
        })
        v_score_mean, ari_mean, ami_mean = \
            v_measure_score(true_labels, pred_labels), \
            adjusted_rand_score(true_labels, pred_labels), \
            adjusted_mutual_info_score(true_labels, pred_labels)
        logging.info("the v_score: {}".format(v_score_mean))
        logging.info("the ami:{}".format(ami_mean))
        logging.info("the ari:{}".format(ari_mean))
        return evaluations

    @staticmethod
    def visualization(umap_data, pred_labels):
        """
        to figure the scatter after umap
        :param pred_labels: the cell labels predicted
        :param umap_data: the data after umap
        :return:
        """
        plt.rc('font', family="Times New Roman", size=13, weight="bold")
        fontdict_label = {'family': "Times New Roman", 'size': 13, 'weight': "bold"}

        colors = []
        for i in range(len(umap_data)):
            color = ["#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])].pop()
            colors.append(color)

        # spines
        ax = plt.axes()
        for sp in ['bottom', 'left']:
            ax.spines[sp].set_color('black')
            ax.spines[sp].set_linewidth('1.5')
        for sp in ['right', 'top']:
            ax.spines[sp].set_visible(False)
            ax.spines[sp].set_visible(False)
        # grid
        ax.grid(axis='both', linestyle='--', alpha=0.4, zorder=0, linewidth=0.8)
        # title and label
        ax.set_title("Louvain Clustering", fontdict_label)
        ax.set_xlabel("UMAP_1", fontdict_label)
        ax.set_ylabel("UMAP_2", fontdict_label)
        for i in range(len(umap_data)):
            plt.scatter(umap_data[i][0], umap_data[i][1], s=10, c=colors[pred_labels[i]])
        plt.show()

