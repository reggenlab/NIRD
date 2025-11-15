__all__ = [
	"MatrixFactorization",
	"PrincipalComponentAnalysis",
	"PrincipalComponentAnalysis_old",
	"SingularValueDecomposition",
	"NonNegativeMatrixFactorization",
	"NonNegativeMatrixFactorization_NIMFA",
	"IteratedConditionalModesNMF",
	"BayesianDecomposition",
	"BinaryMatrixFactorization",
	"FisherNMF",
	"LeastSquaresNMF",
	"KullbackLeiblerDivergenceNMF",
	"EuclideanNMF",
	"ProbabilisticNMF",
	"ProbabilisticSparseNMF",
	"SparseNMF",
	"SparseNetworkRegularizedNMF",
	"PenalizedMatrixFactorization",
	"SeparableNMF",
	"Kernel_PCA",

	"GENIE3",
	"GrnBoost2",
    "ARACNE",
    "MRNET",
    "RELNET",
    "C3NET",

	"str2method"
]

from .MatrixFactorization import MatrixFactorization
from .PrincipalComponentAnalysis import PrincipalComponentAnalysis
from .PrincipalComponentAnalysis_old import PrincipalComponentAnalysis_old
from .SingularValueDecomposition import SingularValueDecomposition
from .NonNegativeMatrixFactorization import NonNegativeMatrixFactorization
from .IteratedConditionalModesNMF import IteratedConditionalModesNMF
from .BayesianDecomposition import BayesianDecomposition
from .BinaryMatrixFactorization import BinaryMatrixFactorization
from .FisherNMF import FisherNMF
from .LeastSquaresNMF import LeastSquaresNMF
from .KullbackLeiblerDivergenceNMF import KullbackLeiblerDivergenceNMF
from .EuclideanNMF import EuclideanNMF
from .ProbabilisticNMF import ProbabilisticNMF
from .ProbabilisticSparseNMF import ProbabilisticSparseNMF
from .SparseNMF import SparseNMF
from .SparseNetworkRegularizedNMF import SparseNetworkRegularizedNMF
from .PenalizedMatrixFactorization import PenalizedMatrixFactorization
from .SeparableNMF import SeparableNMF
from .Kernel_PCA import Kernel_PCA

from .GENIE3 import GENIE3
from .GrnBoost2 import GrnBoost2
from .ARACNE import ARACNE
from .MRNET import MRNET
from .RELNET import RELNET
from .C3NET import C3NET


str2method = {'PCA': PrincipalComponentAnalysis,
              'PrincipalComponentAnalysis': PrincipalComponentAnalysis,
              'SVD': SingularValueDecomposition,
              'SingularValueDecomposition': SingularValueDecomposition,
              'NMF': NonNegativeMatrixFactorization,
              'NonNegativeMatrixFactorization': NonNegativeMatrixFactorization,
              'NIRD': Kernel_PCA,
              'Kernel_PCA': Kernel_PCA,
              'ICM': IteratedConditionalModesNMF,
              'IteratedConditionalModesNMF': IteratedConditionalModesNMF,
              'BD': BayesianDecomposition,
              'BayesianDecomposition': BayesianDecomposition,
              'BMF': BinaryMatrixFactorization,
              'BinaryMatrixFactorization': BinaryMatrixFactorization,
              # 'LFNMF': FisherNMF,
              # 'FisherNMF': FisherNMF,
              'LSNMF': LeastSquaresNMF,
              'LeastSquaresNMF': LeastSquaresNMF,
              'KLD_NMF': KullbackLeiblerDivergenceNMF,
              'KullbackLeiblerDivergenceNMF': KullbackLeiblerDivergenceNMF,
              'ENMF': EuclideanNMF,
              'EuclideanNMF': EuclideanNMF,
              'PMF': ProbabilisticNMF,
              'ProbabilisticNMF': ProbabilisticNMF,
              # 'PSMF': ProbabilisticSparseNMF,
              # 'ProbabilisticSparseNMF': ProbabilisticSparseNMF,
              'SparseNMF': SparseNMF,
              'SNMF': SparseNMF,
              # 'SNMNMF': SparseNetworkRegularizedNMF,
              # 'SparseNetworkRegularizedNMF': SparseNetworkRegularizedNMF,
              'PMFCC': PenalizedMatrixFactorization,
              'PenalizedMatrixFactorization': PenalizedMatrixFactorization,
              'SepNMF': SeparableNMF,
              'SeparableNMF': SeparableNMF,

              'GENIE3': GENIE3,
              'GrnBoost2': GrnBoost2,
              'ARACNE': ARACNE,
              'MRNET': MRNET,
              'RELNET': RELNET,
              'C3NET': C3NET
              }
