
import numpy as np
from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from multiprocessing import Pool
from .MatrixFactorization import MatrixFactorization
import warnings
warnings.filterwarnings('ignore')


class NonNegativeMatrixFactorization(MatrixFactorization):

	def __init__(self, data) -> None:
		super().__init__(data)
		self.regressor = self._get_estimator()
		self.factorizer = self._get_factorizer()
		self.input_idx = [i for i, gene in enumerate(self.gene_names) if gene in self.regulators]

		self.results = None
		self.exp_in_lower_space = None
		self.feature_importances = np.zeros((self.gene_count, self.n_components))
		self.feature_contribution_in_lower_space = None
		self.feature_importance_in_lower_space = None

	@staticmethod
	def _get_estimator():
		# TODO: can include other Tree methods also
		return RandomForestRegressor(n_estimators=50, max_features='sqrt', max_depth=8, verbose=0, random_state=123)

	def _get_factorizer(self):
		return NMF(n_components=self.n_components, init='random', random_state=123)

	def _preprocess(self):
		# standardScaler = StandardScaler().fit(self.exp)
		# temp = standardScaler.fit_transform(self.exp)
		# minMaxScaler = MinMaxScaler().fit(temp)
		# self.data['_scaled_data'] = np.mat(minMaxScaler.fit_transform(temp))
		# return self.data['_scaled_data']
		minMaxScaler = MinMaxScaler().fit(self.exp)
		self.data['_scaled_data'] = np.mat(minMaxScaler.fit_transform(self.exp))
		return self.data['_scaled_data']


	def factorize(self):
		_nmf = self.factorizer
		_nmf.fit(self.scaled_exp)

		self.low_rank_exp = _nmf.transform(self.scaled_exp)
		self.mixture_matrix = abs(_nmf.components_)

	def low_rank_model(self):
		clubbed_ip = [(self.exp, i, self.input_idx, self.exp_in_lower_space, self.feature_contribution_in_lower_space)
		              for i
		              in range(self.gene_count)]

		# Multiprocessing
		pool = Pool(self.n_threads)
		self.fis = pool.map(self.data2network, clubbed_ip)

		for (i, fi) in self.fis:
			self.feature_importances[i, :] = fi

	def data2network(self, args):
		return [args[1], self._data2network(args[0], args[1], args[2], args[3], args[4])]

	def _data2network(self, scaled_exp, i, input_idx, exp_in_lower_space, feature_contribution_in_lower_space):
		X = exp_in_lower_space.copy()
		y = scaled_exp[:, i].copy()
		y = y / np.std(y)
		self.regressor.fit(X, y)
		return self.regressor.feature_importances_.copy()

	def reverse_factorize(self):
		feature_importance_in_lower_space = self.feature_importances
		feature_importance_in_higher_space = np.dot(feature_importance_in_lower_space,
		                                            self.feature_contribution_in_lower_space) / \
		                                     (np.sum(feature_importance_in_lower_space) + 0.0001)
		self.network = feature_importance_in_higher_space




