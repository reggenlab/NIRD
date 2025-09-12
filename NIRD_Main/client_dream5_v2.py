# %%Imports

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from MF_Evaluation import str2eval
from MF_Datasets import str2dataset
from MF_Classes import str2method
from MF_Clients.helper import data2network

import warnings
warnings.filterwarnings('ignore')


# %% States

df_global_matched = pd.DataFrame()
global_net = list()
features = list()

methods = ['NIRD', 'GENIE3', 'GrnBoost2']
datasets = ['dream5_net1']


# %% Datasets

dataset = datasets[0]
data = str2dataset[dataset](dataset_name=dataset).get_dataset()


# %% Methods

method = methods[0]

d_expr = data[0]
d_gold = data[1]

net_expr = data2network(d_expr, method)
net_gold = data2network(d_gold, method)


# %%

