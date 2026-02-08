import numpy as np
from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from multiprocessing import Pool
from .MatrixFactorization import MatrixFactorization
import nimfa
import warnings
warnings.filterwarnings('ignore')


class NonNegativeMatrixFactorization_NIMFA(MatrixFactorization):

	def __init__(self, data) -> None:
		super().__init__(data)
		self.regressor = self._get_estimator()
		self.factorizer = self._get_factorizer()

		self.results = None
		self.low_rank_exp = None
		self.feature_importances = np.zeros((len(self.regulators), self.n_components))
		self.mixture_matrix = None
		self.low_rank_feature_importance = None

	def _get_factorizer(self):
		pass

	@staticmethod
	def _get_estimator():
		# TODO: can include other Tree methods also
		return RandomForestRegressor(n_estimators=50, max_features='sqrt', max_depth=8, verbose=0, random_state=123)

	def log_normalize(self, data, _managed=True):
		data = np.where(data > 0, data, -10)
		return np.log(data, out=data, where=data > 0)

	def _preprocess(self):
		minMaxScaler = MinMaxScaler().fit(self.exp) # Non-negativity condition
		self.data['_scaled_data'] = np.mat(minMaxScaler.fit_transform(self.exp))
		return self.data['_scaled_data']

	def _binarize(self, x, cutoff):
		return np.digitize(x=x, bins=[cutoff])

	def factorize(self):
		_nmf, _nmf_fit = self.factorizer
		self.low_rank_exp = _nmf.W
		self.mixture_matrix = _nmf.H

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
		X = low_rank_exp.copy()
		y = scaled_exp[:, i].copy()
		# y = y / np.std(y.astype('float32'))
		self.regressor.fit(X, y)
		return self.regressor.feature_importances_

	def reverse_factorize(self):
		self.low_rank_feature_importance = self.feature_importances
		true_rank_feature_importance = np.dot(self.low_rank_feature_importance, self.mixture_matrix) / \
		                               (np.sum(self.low_rank_feature_importance) + 0.0001)
		self.network = true_rank_feature_importance
