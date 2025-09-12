import pandas as pd
import numpy as np

from MF_Clients.helper import do_feature_selection
from MF_Datasets import MF_Datasets


class Dataset_mESC(MF_Datasets):

	def __init__(self, dataset_name):
		super().__init__(dataset_name)


	def get_dataset(self):
		try:
			drop_seq = self.csv2dict(data_filepath=self.dataset_path + '/dropSeq_noisy.csv')
			smart_seq = self.csv2dict(data_filepath=self.dataset_path + '/smartSeq_noisy.csv')
		except:
			drop_seq, smart_seq = self.prepare_data(data_filepath=self.dataset_path + '/fpkm_all.csv')
			drop_seq = self.csv2dict(data_filepath=self.dataset_path + '/dropSeq_noisy.csv')
			smart_seq = self.csv2dict(data_filepath=self.dataset_path + '/smartSeq_noisy.csv')

		finally:
			return [drop_seq, smart_seq]


	def prepare_data(self, data_filepath):
		d = pd.read_csv(data_filepath, index_col=0)
		col = list(d.columns)
		col_drop = [i for i in col if 'DropSeq' in i]
		col_smart = [i for i in col if 'SmartSeq' in i]

		dropSeq = d[col_drop]
		smartSeq = d[col_smart]

		dropSeq = do_feature_selection(dropSeq.T)  # Sample * features (genes)
		smartSeq = do_feature_selection(smartSeq.T)

		intersected_features = set(dropSeq.columns) & set(smartSeq.columns)
		dropSeq = dropSeq[intersected_features]
		smartSeq = smartSeq[intersected_features]

		dropSeq.to_csv('../MF_Datasets/mESC/dropSeq_noisy.csv')
		smartSeq.to_csv('../MF_Datasets/mESC/smartSeq_noisy.csv')

		return dropSeq, smartSeq


	def csv2df(self, data_filepath):
		return pd.read_csv(filepath_or_buffer=data_filepath, index_col=0)

	def csv2array(self, tf_filepath):
		if tf_filepath is None:
			return None
		else:
			with open(tf_filepath) as file:
				return np.array(line.strip() for line in file.readlines())



