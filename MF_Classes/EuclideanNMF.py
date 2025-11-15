
from .NonNegativeMatrixFactorization_NIMFA import NonNegativeMatrixFactorization_NIMFA
import nimfa
import warnings
warnings.filterwarnings('ignore')


class EuclideanNMF(NonNegativeMatrixFactorization_NIMFA):

	def __init__(self, data) -> None:
		super().__init__(data)

	def _get_factorizer(self):
		# Seedings: https://nimfa.biolab.si/nimfa.methods.seeding.html
		_nmf = nimfa.Nmf(V=self.scaled_exp, seed="random", rank=self.n_components, update='euclidean')
		_fit = _nmf()
		return _nmf, _fit

	def _preprocess(self):
		self.data['_scaled_data'] = self.exp # No scaling or normalization
		return self.data['_scaled_data']