import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from .MatrixFactorization import MatrixFactorization
import warnings
warnings.filterwarnings('ignore')


class PrincipalComponentAnalysis_old(MatrixFactorization):
    # ref. https: // stats.stackexchange.com / a / 229093 / 143495
    def __init__(self, data) -> None:
        super().__init__(data)
        self.n_pc = 10
        self.regressor = self._get_estimator()
        self.input_idx = [i for i, gene in enumerate(self.gene_names) if gene in self.regulators]
        self.network = np.zeros((self.gene_count, self.gene_count))

        self.results = None
        self.exp_in_pca_space = None
        self.feature_importances = None
        self.feature_contribution_in_pca_space = None
        self.feature_importance_in_pca_space = None

    @staticmethod
    def _get_estimator():
        # TODO: can include other Tree methods also
        return RandomForestRegressor(n_estimators=50, max_features='sqrt', max_depth=8, verbose=0)

    def _preprocess(self):
        # scaler = StandardScaler().fit(self.exp)
        # self.data['_scaled_data'] = np.mat(scaler.fit_transform(self.data['_data']))
        # return self.data['_scaled_data']
        return self.data['_data']

    def prepare_data(self):
        self.exp = self.data['_data']
        self.gene_names = self.data['_features']
        self.gene_count = self.data['_feat_count']
        self.tf_names = self.data['_tf_names']

        if self.tf_names is None:
            self.regulators = list(set(self.gene_names))
        else:
            self.regulators = list(set(self.gene_names) & set(self.tf_names))

        self.input_idx = [i for i, gene in enumerate(self.gene_names) if gene in self.regulators]
        self.network = np.zeros((self.gene_count, self.gene_count))

        self._preprocess()

    def factorize(self):
        _pca = PCA(n_components=10, random_state=123)
        _pca.fit(self.scaled_exp)
        self.exp_in_pca_space = _pca.transform(self.scaled_exp)
        self.feature_contribution_in_pca_space = abs(_pca.components_)


    def _preprocess(self):
        scaler = StandardScaler().fit(self.exp)
        self.data['_scaled_data'] = np.mat(scaler.fit_transform(self.data['_data']))
        return self.data['_scaled_data']

    def low_rank_model(self):
        input_data = list()
        for i in range(self.gene_count):
            input_data.append([self.scaled_exp, i, self.input_idx, self.exp_in_pca_space, self.feature_contribution_in_pca_space])

        # Multiprocessing
        pool = Pool(self.n_threads)
        results = pool.map(self.data2network, input_data)

        # for sequential testing
        # for ip in input_data:
        #     results = self.data2network(ip)

        for (i, vi) in results:
            self.network[i, :] = vi

    def data2network(self, args):
        return [args[1], self._data2network(args[0], args[1], args[2], args[3], args[4])]

    def _data2network(self, exp, output_idx, input_idx, exp_in_pca_space, feature_contribution_in_pca_space):

        input_idx = input_idx[:]

        X = exp_in_pca_space
        y = exp[:, output_idx]
        y = y / np.std(y)
        # y = (y - np.mean(y)) / np.std(y)

        rf_regressor = RandomForestRegressor(n_estimators=50, max_features='sqrt', max_depth=8, verbose=0, random_state=123)
        rf_regressor.fit(X, y)

        # Compute importance scores
        feature_importance_in_pca_space = rf_regressor.feature_importances_
        feature_importance_in_gene_space = np.dot(feature_importance_in_pca_space, feature_contribution_in_pca_space) / \
                                           (np.sum(feature_importance_in_pca_space) + 0.0001)

        row = np.zeros(self.data['_feat_count'])
        row[input_idx] = feature_importance_in_gene_space
        return row

    def reverse_factorize(self):
        pass

    def fit(self):
        self.factorize()
        self.low_rank_model()
        return self.network