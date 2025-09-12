
import pandas as pd
import numpy as np

from MF_Clients.helper import do_feature_selection
from MF_Datasets import MF_Datasets


class Double_Expr(MF_Datasets):

    def __init__(self, dataset_name, file1=None, file2=None):
        super().__init__(dataset_name)
        self.file1 = file1
        self.file2 = file2

    def get_dataset(self):
        try:
            expr1 = self.csv2dict(data_filepath=self.file1)
            expr2 = self.csv2dict(data_filepath=self.file2)
        except:
            expr1, expr2 = self.prepare_data(self.file1, self.file2)
            expr1 = self.csv2dict(data_filepath=self.file1)
            expr2 = self.csv2dict(data_filepath=self.file2)
        finally:
            return [expr1, expr2]

    def prepare_data(self, file1, file2):
        d1 = pd.read_csv(file1, index_col=0)
        d2 = pd.read_csv(file2, index_col=0)

        col_expr1 = list(d1.columns)
        col_expr2 = list(d2.columns)

        expr1 = d1[col_expr1]
        expr2 = d2[col_expr2]

        # Apply feature selection
        expr1 = do_feature_selection(expr1.T)  # Sample * features (genes)
        expr2 = do_feature_selection(expr2.T)

        # Keep only intersecting genes
        intersected_features = set(expr1.columns) & set(expr2.columns)
        expr1 = expr1[list(intersected_features)]
        expr2 = expr2[list(intersected_features)]

        # Save processed files back
        expr1.to_csv(self.file1)
        expr2.to_csv(self.file2)

        return expr1, expr2

    def csv2df(self, data_filepath):
        return pd.read_csv(filepath_or_buffer=data_filepath, index_col=0)

    def csv2array(self, tf_filepath):
        if tf_filepath is None:
            return None
        else:
            with open(tf_filepath) as file:
                return np.array([line.strip() for line in file.readlines()])
