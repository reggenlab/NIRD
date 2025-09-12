
import pandas as pd
import numpy as np

from MF_Clients.helper import do_feature_selection
from MF_Datasets import MF_Datasets

class Gold_Data(MF_Datasets):

    def __init__(self, dataset_name, expr_file=None, tf_file=None, gold_file=None):
        super().__init__(dataset_name)
        self.expr_file = expr_file
        self.tf_file = tf_file
        self.gold_file = gold_file

    def get_dataset(self):
        """
        Returns two datasets:
        1. Expression data + TF list (dict)
        2. Gold standard network (dict)
        """
        try:
            net1_exp = self.csv2dict(
                data_filepath=self.expr_file,
                tf_filepath=self.tf_file
            )
            net1_gold = self.gold2dict(gold_filepath=self.gold_file)
        except Exception as e:
            raise FileNotFoundError(f"Error loading dataset: {e}")

        return [net1_exp, net1_gold]

    def csv2df(self, data_filepath):
        """Load tab-separated expression data"""
        return pd.read_csv(filepath_or_buffer=data_filepath, sep='\t', header=0)

    def csv2array(self, tf_filepath):
        """Load TF list as numpy array"""
        if tf_filepath is None:
            return None
        else:
            with open(tf_filepath) as file:
                return np.array([line.strip() for line in file.readlines()])
