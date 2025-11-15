from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class MatrixFactorization(ABC):

    def __init__(self, data) -> None:
        self.n_threads = 8
        self.n_components = 30
        # self.n_components = min(30, len(data['_data']), len(self.gene_names))
        self.data = data
        self.gene_names = data['_features']
        self.gene_count = data['_feat_count']
        self.tf_names = data['_tf_names']

        self.regulators = self.get_regulators()
        self.network = np.zeros((len(self.regulators), len(self.regulators)))

        self.exp = self.get_exp_from_regulators()
        self.scaled_exp = self._preprocess()

    def __str__(self) -> str:
        return "MF Object for data having size {}".format(self.data.shape)

    def __repr__(self) -> str:
        return super().__repr__()

    def get_regulators(self):

        if self.tf_names is None:
            regulators = list(set(self.gene_names))
        else:
            regulators = list(set(self.gene_names) & set(self.tf_names))
        return regulators

    def get_exp_from_regulators(self):
        self.input_idx = [i for i, gene in enumerate(self.gene_names) if gene in self.regulators]
        return self.data['_data'][:, self.input_idx]

    def quantile_normalize(self, df_input):  # df_input -> genes*samples
        # from https://stackoverflow.com/a/43260153/4467129
        df = df_input.copy()
        df = pd.DataFrame(df)
        dic = {}
        for col in df:
            dic.update({col: sorted(df[col])})
        sorted_df = pd.DataFrame(dic)
        rank = sorted_df.mean(axis=1).tolist()
        for col in df:
            t = np.searchsorted(np.sort(df[col]), df[col])
            df[col] = [rank[i] for i in t]
        return df.values

    def _remove_loop(self, adj):
        return adj - np.diag(np.diag(adj))

    @abstractmethod
    def factorize(self):
        pass

    @abstractmethod
    def _preprocess(self):
        pass

    @abstractmethod
    def low_rank_model(self):
        pass

    @abstractmethod
    def reverse_factorize(self):
        pass

    def fit(self):
        self.factorize()
        self.low_rank_model()
        self.reverse_factorize()

        network_without_loop = self._remove_loop(self.network)
        return network_without_loop, self.regulators