
from .NonNegativeMatrixFactorization_NIMFA import NonNegativeMatrixFactorization_NIMFA
import nimfa
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class BinaryMatrixFactorization(NonNegativeMatrixFactorization_NIMFA):

	def __init__(self, data) -> None:
		super().__init__(data)

	def _get_factorizer(self):
		_nmf = nimfa.Bmf(V=self.scaled_exp, rank=self.n_components)
		_fit = _nmf()
		return _nmf, _fit

	def _preprocess(self):
		temp = self.log_normalize(self.exp).copy()
		self.data['_scaled_data'] = np.transpose(np.matrix([self._binarize(x=temp[:, i], cutoff=(np.min(temp)+np.max(temp))*.25).flatten() for i in range(temp.shape[1])]))
		return self.data['_scaled_data']
