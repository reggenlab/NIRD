import pandas as pd
import numpy as np

from MF_Clients.helper import do_feature_selection
from MF_Datasets import MF_Datasets


################################# For pancreas data #######################################

# class Human_Knee_Cartilage(MF_Datasets):

#     def __init__(self, dataset_name):
#         super().__init__(dataset_name)

#     def get_dataset(self):
#         try:
#             old_expression = self.csv2dict(data_filepath=self.dataset_path + '/pancreas_old_beta.csv')
#             young_expression = self.csv2dict(data_filepath=self.dataset_path + '/pancreas_young_beta.csv')
#         except:
#             old_expression, young_expression = self.prepare_data(data_filepath=self.dataset_path + '/pancreas_beta.csv')
#             old_expression = self.csv2dict(data_filepath=self.dataset_path + '/pancreas_old_beta.csv')
#             young_expression = self.csv2dict(data_filepath=self.dataset_path + '/pancreas_young_beta.csv')

#         finally:
#             return [old_expression, young_expression]

#     def prepare_data(self, data_filepath):
#         d = pd.read_csv(data_filepath, index_col=0)
#         col = list(d.columns)
#         col_old = [i for i in col if 'GSM' in i]
#         col_young = [i for i in col if 'GSM' in i]

#         old_expression = d[col_old]
#         young_expression = d[col_young]

#         old_expression = do_feature_selection(old_expression.T)  # Sample * features (genes)
#         young_expression = do_feature_selection(young_expression.T)

#         intersected_features = set(old_expression.columns) & set(young_expression.columns)
#         old_expression = old_expression[intersected_features]
#         young_expression = young_expression[intersected_features]

#         old_expression.to_csv('../MF_Datasets/human_knee_cartilage/pancreas_old_beta.csv')
#         young_expression.to_csv('../MF_Datasets/human_knee_cartilage/pancreas_young_beta.csv')

#         return old_expression, young_expression

#     def csv2df(self, data_filepath):
#         return pd.read_csv(filepath_or_buffer=data_filepath, index_col=0)

#     def csv2array(self, tf_filepath):
#         if tf_filepath is None:
#             return None
#         else:
#             with open(tf_filepath) as file:
#                 return np.array(line.strip() for line in file.readlines())



############################### For human knee cartilage data ################################

class Human_Knee_Cartilage(MF_Datasets):

    def __init__(self, dataset_name):
        super().__init__(dataset_name)

    def get_dataset(self):
        try:
            control_expression = self.csv2dict(data_filepath=self.dataset_path + '/control_HTC_expression_data.csv')
            noncontrol_expression = self.csv2dict(data_filepath=self.dataset_path + '/noncontrol_HTC_expression_data.csv')
        except:
            control_expression, noncontrol_expression = self.prepare_data(data_filepath=self.dataset_path + '/sc_counts.csv')
            control_expression = self.csv2dict(data_filepath=self.dataset_path + '/control_HTC_expression_data.csv')
            noncontrol_expression = self.csv2dict(data_filepath=self.dataset_path + '/noncontrol_HTC_expression_data.csv')

        finally:
            return [control_expression, noncontrol_expression]

    def prepare_data(self, data_filepath):
        d = pd.read_csv(data_filepath, index_col=0)
        col = list(d.columns)
        col_control = col.iloc[:, 0]
        col_noncontrol = col.iloc[:, 0]

        control_expression = d[col_control]
        noncontrol_expression = d[col_noncontrol]

        control_expression = do_feature_selection(control_expression.T)  # Sample * features (genes)
        noncontrol_expression = do_feature_selection(noncontrol_expression.T)

        intersected_features = set(control_expression.columns) & set(noncontrol_expression.columns)
        control_expression = control_expression[intersected_features]
        noncontrol_expression = noncontrol_expression[intersected_features]

        control_expression.to_csv('../MF_Datasets/human_knee_cartilage/control_HTC_expression_data.csv')
        noncontrol_expression.to_csv('../MF_Datasets/human_knee_cartilage/noncontrol_HTC_expression_data.csv')

        return control_expression, noncontrol_expression

    def csv2df(self, data_filepath):
        return pd.read_csv(filepath_or_buffer=data_filepath, index_col=0)

    def csv2array(self, tf_filepath):
        if tf_filepath is None:
            return None
        else:
            with open(tf_filepath) as file:
                return np.array(line.strip() for line in file.readlines())