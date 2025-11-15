
from .NonNegativeMatrixFactorization import NonNegativeMatrixFactorization
import nimfa
import warnings
warnings.filterwarnings('ignore')

class ProbabilisticSparseNMF(NonNegativeMatrixFactorization):

	def __init__(self, data) -> None:
		super().__init__(data)

	def _get_factorizer(self):
		_nmf = nimfa.Psmf(V=self.scaled_exp, rank=self.n_components)
		_fit = _nmf()
		return _nmf, _fit

	def _preprocess(self):
		self.data['_scaled_data'] = self.data['_data'] # No scaling or normalization
		return self.data['_scaled_data']

