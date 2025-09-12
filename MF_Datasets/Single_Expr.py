import pandas as pd
import numpy as np
import shutil
import os

from MF_Clients.helper import do_feature_selection
from MF_Datasets import MF_Datasets

class Single_Expr(MF_Datasets):

    def __init__(self, dataset_name, file1=None):
        super().__init__(dataset_name)
        self.file1 = file1

    def get_dataset(self):
        # Prepare data if needed
        try:
            expr1 = self.csv2dict(data_filepath=self.file1)
            expr2 = self.csv2dict(data_filepath=self.file1)  # same file used twice
        except:
            expr1, expr2 = self.prepare_data(self.file1)
            expr1 = self.csv2dict(data_filepath=self.file1)
            expr2 = self.csv2dict(data_filepath=self.file1)
        finally:
            return [expr1, expr2]

    def prepare_data(self, file1):
        d = pd.read_csv(file1, index_col=0)

        # Keep all columns (genes)
        col_expr = list(d.columns)
        expr = d[col_expr]

        # Apply feature selection
        expr = do_feature_selection(expr.T)  # Samples * features (genes)
        expr.to_csv(file1)

        # Since we are using the same file twice, copy it
        file2 = file1.replace(".csv", "_copy.csv")
        shutil.copy(file1, file2)

        return expr, expr

    def csv2df(self, data_filepath):
        return pd.read_csv(filepath_or_buffer=data_filepath, index_col=0)

    def csv2array(self, tf_filepath):
        if tf_filepath is None:
            return None
        else:
            with open(tf_filepath) as file:
                return np.array([line.strip() for line in file.readlines()])
