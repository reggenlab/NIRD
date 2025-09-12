import pandas as pd
import numpy as np

from MF_Clients.helper import do_feature_selection
from MF_Datasets import MF_Datasets


class Dataset_transcription_velocity(MF_Datasets):

	def __init__(self, dataset_name):
		super().__init__(dataset_name)


	def get_dataset(self):
		try:
			zeroth_hr = self.csv2dict(data_filepath=self.dataset_path + '/0th_hr_endo_RNA_Velo.csv')
			zeroth_hr_expr = self.csv2dict(data_filepath=self.dataset_path + '/12th_hr_endo_RNA_Velo.csv')
		except:
			zeroth_hr, zeroth_hr_expr = self.prepare_data(data_filepath=self.dataset_path + '/GSE75748_sc_time_course_ec.csv')
			zeroth_hr = self.csv2dict(data_filepath=self.dataset_path + '/0th_hr_endo_RNA_Velo.csv')
			zeroth_hr_expr = self.csv2dict(data_filepath=self.dataset_path + '/12th_hr_endo_RNA_Velo.csv')

		finally:
			return [zeroth_hr, zeroth_hr_expr]


	def prepare_data(self, data_filepath):
		d = pd.read_csv(data_filepath, index_col=0)
		col = list(d.columns)
		col_zeroth = [i for i in col if 'SRR' in i]
		col_zeroth_expr = [i for i in col if 'SRR' in i]

		zeroth_hr_data = d[col_zeroth]
		zeroth_hr_expr_data = d[col_zeroth_expr]

		zeroth_hr_data = do_feature_selection(zeroth_hr_data.T)  # Sample * features (genes)
		zeroth_hr_expr_data = do_feature_selection(zeroth_hr_expr_data.T)

		intersected_features = set(zeroth_hr_data.columns) & set(zeroth_hr_expr_data.columns)
		zeroth_hr_data = zeroth_hr_data[intersected_features]
		zeroth_hr_expr_data = zeroth_hr_expr_data[intersected_features]

		zeroth_hr_data.to_csv('../MF_Datasets/transcription_velocity/0th_hr_endo_RNA_Velo.csv')
		zeroth_hr_expr_data.to_csv('../MF_Datasets/transcription_velocity/12th_hr_endo_RNA_Velo.csv')

		return zeroth_hr_data, zeroth_hr_expr_data


	def csv2df(self, data_filepath):
		return pd.read_csv(filepath_or_buffer=data_filepath, index_col=0)

	def csv2array(self, tf_filepath):
		if tf_filepath is None:
			return None
		else:
			with open(tf_filepath) as file:
				return np.array(line.strip() for line in file.readlines())

