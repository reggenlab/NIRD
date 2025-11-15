
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD, KernelPCA
from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
from .MatrixFactorization import MatrixFactorization
import warnings
warnings.filterwarnings('ignore')


class Kernel_PCA(MatrixFactorization):

	def __init__(self, data) -> None:
		super().__init__(data)
		self.regressor = self._get_estimator()
		self.factorizer = self._get_factorizer()

		self.results = None
		self.low_rank_exp = None
		self.feature_importances = np.zeros((len(self.regulators), self.n_components))
		self.mixture_matrix = None
		self.low_rank_feature_importance = None


	@staticmethod
	def _get_estimator():
		# TODO: can include other Tree methods also
		return RandomForestRegressor(n_estimators=50, max_features='sqrt', max_depth=8, verbose=0, random_state=123)
		# return ExtraTreesRegressor(n_estimators=1000)

	def _get_factorizer(self):
		return KernelPCA(n_components=self.n_components, kernel='linear')

	def _preprocess(self):
		# scaler = StandardScaler().fit(self.exp)
		# self.data['_scaled_data'] = np.mat(scaler.fit_transform(self.exp))
		# return self.data['_scaled_data']
		return self.exp

	def factorize(self):
		_kpca = self.factorizer
		_kpca.fit(self.scaled_exp)

		self.low_rank_exp = _kpca.transform(self.scaled_exp)
		self.mixture_matrix = self.get_feature_contribution()

	def get_feature_contribution(self):
		df1 = pd.DataFrame(self.low_rank_exp)
		df2 = pd.DataFrame(self.scaled_exp)
		df = pd.concat([df2, df1], axis=1)
		daret = df.corr().iloc[len(self.regulators):, :len(self.regulators)]
		return daret

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
		# true_rank_feature_importance = np.dot(self.low_rank_feature_importance, self.mixture_matrix) / \
		#                                   (np.sum(self.low_rank_feature_importance) + 0.0001)
		true_rank_feature_importances = np.zeros((len(self.regulators), len(self.regulators)))
		for i in range(1,len(self.regulators)):
			for j in range(1,len(self.regulators)):
				t1 = self.low_rank_feature_importance[ i,:]
				t2 = self.mixture_matrix.iloc[:,i]
				true_rank_feature_importances[i,j] = np.dot(t1, abs(t2))/(np.sum(abs(t2)) + 0.0001) 

		#true_rank_feature_importance = np.dot(self.low_rank_feature_importance, abs(self.mixture_matrix)) / \s
		#                                  (np.sum(abs(self.mixture_matrix)) + 0.0001)
		
		self.network = true_rank_feature_importances




