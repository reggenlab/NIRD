from .MF_Datasets import MF_Datasets
import pandas as pd
import numpy as np


class Dataset_Dream5_Net4(MF_Datasets):

	def __init__(self, dataset_name):
		super().__init__(dataset_name)
		self.dataset_path = '../MF_Datasets/dream5' + '/net4'

	def get_dataset(self):
		net4_exp = self.csv2dict(data_filepath=self.dataset_path + '/dream5_net4_expression_data_qnorm.tsv',
		                    tf_filepath=self.dataset_path + '/dream5_net4_transcription_factors.tsv')
		net4_gold = self.gold2dict(gold_filepath=self.dataset_path + '/dream5_net4_gold.tsv')
		return [net4_exp, net4_gold]

	def csv2df(self, data_filepath):
		return pd.read_csv(filepath_or_buffer=data_filepath, sep='\t', header=0)

	def csv2array(self, tf_filepath):
		if tf_filepath is None:
			return None
		else:
			with open(tf_filepath) as file:
				return np.array([line.strip() for line in file.readlines()])
