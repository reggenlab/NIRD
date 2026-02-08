
import pandas as pd
from arboreto.algo import grnboost2
from distributed import Client
from .grnboost2_original import GRNBoost2_


class GrnBoost2(object):

	def __init__(self, data):
		self.data = data
		self.regulators = self.get_regulators()

	def get_regulators(self):
		if self.data['_tf_names'] is None:
			regulators = list(set(self.data['_features']))
		else:
			regulators = list(set(self.data['_features']) & set(self.data['_tf_names']))
		return regulators


	# def fit(self):
	# 	inDF = pd.DataFrame(data=self.data['_data'], columns=self.data['_features'])
	# 	inDF = inDF.loc[:,self.regulators]
	# 	client = Client(processes=False)
	# 	# print(client.dashboard_link)
	# 	network = grnboost2(expression_data=inDF.to_numpy(),
	# 	                 client_or_address=client,
	# 	                 gene_names=inDF.columns)
	# 	return network.values, self.regulators
	

	def fit(self):
		inDF = pd.DataFrame(data=self.data['_data'], columns=self.data['_features'])
		inDF = inDF.loc[:,self.regulators]

		network = GRNBoost2_(expr_data=inDF.to_numpy(),
		                 gene_names=list(inDF.columns),
		                 nthreads=40)
		return network, self.regulators
	

