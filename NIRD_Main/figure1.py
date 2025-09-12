# %% Imports
import pandas as pd

# %% Reading the data
f_name1 = "./Beeline/inputs/example/GSD/GRNVBEM/ExpressionData0.csv"
f_name2 = "./Beeline/inputs/example/GSD/GRNVBEM/ExpressionData0.csv"
ex1 = pd.read_csv(f_name1, sep=',', index_col=0).T
ex2 = pd.read_csv(f_name2, sep=',', index_col=0).T

# %%








