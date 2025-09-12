import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

class MF_Datasets(ABC):

	def __init__(self, dataset_name):
		self.dataset_name = dataset_name
		self.dataset_path = '../MF_Datasets/' + self.dataset_name

	def csv2dict(self, data_filepath, tf_filepath=None):
		data = self.csv2df(data_filepath)
		tfs = self.csv2array(tf_filepath)
		dict = {'_label': data_filepath.strip().split('/')[-1].split('.')[0],
		        '_p_label': data_filepath.strip().split('/')[-2],
		        '_data': np.mat(data.values),
		        '_index': data.index.values,
		        '_features': data.columns.values,
		        '_feat_count': len(data.columns),
		        '_tf_names': tfs,
		        }
		return dict

	def gold2dict(self, gold_filepath):
		data = self.csv2df(gold_filepath)
		dict = {'_label': 'gold',
		        '_edges': np.array(data.values),
		        }
		return dict

	@abstractmethod
	def get_dataset(self):
		pass

	@abstractmethod
	def csv2df(self, data_filepath):
		pass

	@abstractmethod
	def csv2array(self, tf_filepath):
		pass
