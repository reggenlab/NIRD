
from .NonNegativeMatrixFactorization_NIMFA import NonNegativeMatrixFactorization_NIMFA
import nimfa
import warnings
warnings.filterwarnings('ignore')

class SeparableNMF(NonNegativeMatrixFactorization_NIMFA):

	def __init__(self, data) -> None:
		super().__init__(data)

	def _get_factorizer(self):
		_nmf = nimfa.SepNmf(V=self.scaled_exp, seed='random_vcol', rank=self.n_components)
		_fit = _nmf()
		return _nmf, _fit

	def _preprocess(self):
		self.data['_scaled_data'] = self.exp # No scaling or normalization
		return self.data['_scaled_data']