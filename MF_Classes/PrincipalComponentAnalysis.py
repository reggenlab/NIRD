import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import ExtraTreesRegressor

from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler

from .MatrixFactorization import MatrixFactorization
import warnings
warnings.filterwarnings('ignore')


class PrincipalComponentAnalysis(MatrixFactorization):

	# ref. https://stats.stackexchange.com/a/229093/143495

	def __init__(self, data) -> None:
		super().__init__(data)
		self.regressor = self._get_estimator()

		self.results = None
		self.low_rank_exp = None
		self.feature_importances = np.zeros((len(self.regulators), self.n_components))
		self.mixture_matrix = None
		self.low_rank_feature_importance = None

	@staticmethod
	def _get_estimator():
		# TODO: include other Tree methods also
		# return RandomForestRegressor(random_state=123)
		# return ExtraTreesRegressor(n_estimators=1000)
		return RandomForestRegressor(n_estimators=5, max_features='sqrt', max_depth=8, verbose=0, random_state=123)

	def _preprocess(self):
		return self.exp

	def factorize(self):
		_pca = PCA(svd_solver='full', n_components=self.n_components, random_state=123)
		_pca.fit(self.scaled_exp)
		self.low_rank_exp = _pca.transform(self.scaled_exp)
		self.mixture_matrix = abs(_pca.components_)

	def low_rank_model(self):
		clubbed_ip = [(self.exp, i, self.input_idx, self.low_rank_exp, self.mixture_matrix) for i in self.input_idx]

		# Multiprocessing
		pool = Pool(self.n_threads)
		self.fis = pool.map(self.data2network, clubbed_ip)

		for (i, fi) in self.fis:
			self.feature_importances[i, :] = fi

	def data2network(self, args):
		return [args[1], self._data2network(args[0], args[1], args[3])]

	def _data2network(self, scaled_exp, i, low_rank_exp):
		X = low_rank_exp
		y = scaled_exp[:, i]
		y = y / np.std(y)
		self.regressor.fit(X, y)
		return self.regressor.feature_importances_

	def reverse_factorize(self):

		self.low_rank_feature_importance = self.feature_importances
		# true_rank_feature_importance = np.dot(self.low_rank_feature_importance, self.mixture_matrix) / \
		#                                    (np.sum(self.low_rank_feature_importance) + 0.0001)
		true_rank_feature_importance = np.dot(self.low_rank_feature_importance, abs(self.mixture_matrix)) / \
		                                   (np.sum(abs(self.mixture_matrix)) + 0.0001)
		self.network = true_rank_feature_importance



# import warnings
# import numpy as np
# import pandas as pd
# from sklearn.decomposition import PCA
# from sklearn.ensemble import RandomForestRegressor
# from multiprocessing import Pool
# from sklearn.preprocessing import StandardScaler

# from .MatrixFactorization import MatrixFactorization
# warnings.filterwarnings('ignore')


# class PrincipalComponentAnalysis(MatrixFactorization):

#     # ref. https://stats.stackexchange.com/a/229093/143495

#     def __init__(self, data) -> None:
#         super().__init__(data)
#         self.regressor = self._get_estimator()

#         # Ensure n_components does not exceed the feature space
#         self.n_components = min(self.n_components, self.scaled_exp.shape[1])

#         self.results = None
#         self.low_rank_exp = None
#         self.feature_importances = np.zeros((len(self.regulators), self.n_components))
#         self.mixture_matrix = None
#         self.low_rank_feature_importance = None

#     @staticmethod
#     def _get_estimator():
#         # RandomForestRegressor with fixed parameters
#         return RandomForestRegressor(random_state=123)

#     def _preprocess(self):
#         return self.exp

#     def factorize(self):
#         # Ensure PCA uses the adjusted n_components
#         _pca = PCA(svd_solver='full', n_components=self.n_components, random_state=123)
        
#         # Validate the number of components again
#         if self.scaled_exp.shape[1] < self.n_components:
#             raise ValueError(f"n_components={self.n_components} exceeds available features ({self.scaled_exp.shape[1]})")

#         _pca.fit(self.scaled_exp)
#         self.low_rank_exp = _pca.transform(self.scaled_exp)
#         self.mixture_matrix = abs(_pca.components_)

#     def low_rank_model(self):
#         clubbed_ip = [(self.exp, i, self.input_idx, self.low_rank_exp, self.mixture_matrix) 
#                       for i in self.input_idx if i < self.scaled_exp.shape[1]]

#         # Multiprocessing
#         pool = Pool(self.n_threads)
#         self.fis = pool.map(self.data2network, clubbed_ip)

#         for (i, fi) in self.fis:
#             self.feature_importances[i, :] = fi

#     def data2network(self, args):
#         return [args[1], self._data2network(args[0], args[1], args[3])]

#     def _data2network(self, scaled_exp, i, low_rank_exp):
#         # Prevent out-of-bound indexing
#         if i >= scaled_exp.shape[1]:
#             raise IndexError(f"Index {i} out of bounds for axis 1 with size {scaled_exp.shape[1]}")

#         X = low_rank_exp
#         y = scaled_exp[:, i]
#         y = y / np.std(y)
#         self.regressor.fit(X, y)
#         return self.regressor.feature_importances_

#     def reverse_factorize(self):
#         self.low_rank_feature_importance = self.feature_importances
#         true_rank_feature_importance = np.dot(
#             self.low_rank_feature_importance, self.mixture_matrix
#         ) / (np.sum(self.low_rank_feature_importance) + 0.0001)

#         self.network = true_rank_feature_importance
