import pandas as pd
import numpy as np
from arboreto.algo import genie3
from distributed import Client
from .GENIE3_original import GENIE3_


class GENIE3(object):

	def __init__(self, data):
		self.data = data
		self.regulators = self.get_regulators()
		# self._preprocess()

	def get_regulators(self):
		if self.data['_tf_names'] is None:
			regulators = list(set(self.data['_features']))
		else:
			regulators = list(set(self.data['_features']) & set(self.data['_tf_names']))
		return regulators

	def _preprocess(self):
		self.data['_data'] = self.quantile_normalize(self.data['_data'].T).T

	def quantile_normalize(self, df_input):  # df_input -> genes*samples
		# from https://stackoverflow.com/a/43260153/4467129
		df = df_input.copy()
		df = pd.DataFrame(df)
		dic = {}
		for col in df:
			dic.update({col: sorted(df[col])})
		sorted_df = pd.DataFrame(dic)
		rank = sorted_df.mean(axis=1).tolist()
		for col in df:
			t = np.searchsorted(np.sort(df[col]), df[col])
			df[col] = [rank[i] for i in t]
		return df.values

	# Original GENIE3
	def fit(self):
		inDF = pd.DataFrame(data=self.data['_data'], columns=self.data['_features'])
		inDF = inDF.loc[:,self.regulators]

		network = GENIE3_(expr_data=inDF.to_numpy(),
		                 gene_names=list(inDF.columns),
		                 nthreads=40)
		return network, self.regulators

	# Arberato based GENIE3
	# def fit(self):
	# 	inDF = pd.DataFrame(data=self.data['_data'], columns=self.data['_features'])
	# 	inDF = inDF.loc[:, self.regulators]
	# 	client = Client(processes=False)
	# 	# print(client.dashboard_link)
	# 	network = genie3(expression_data=inDF.to_numpy(),
	# 	                 gene_names=list(inDF.columns),
	# 	                 client_or_address=client)
	# 	return network.values, self.regulators
