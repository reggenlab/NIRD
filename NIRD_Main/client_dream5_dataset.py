import pandas as pd
import numpy as np

from MF_Evaluation import str2eval
from MF_Datasets import str2dataset
from MF_Classes import str2method
import seaborn as sns
import matplotlib.pyplot as plt


from MF_Clients.helper import data2network

# %% Main


if __name__ == '__main__':

    df_global_matched = pd.DataFrame()
    global_net = list()
    features = list()

    datasets = ['dream5_net1']#, 'dream5_net2', 'dream5_net3', 'dream5_net4']
    for dataset in datasets:
        data = str2dataset[dataset](dataset_name=dataset).get_dataset()
        print(dataset)
        features = data[0]['_features']
        # methods = list(str2method.keys())[::2]
        methods = ['NIRD', 'GENIE3', 'GrnBoost2']
        # methods = ['NIRD', 'PCA']
        for method in methods:
            networks = [data2network(data=d, method=method) for d in data]
            global_net.append(networks[0]['_corr'])

            # evaluations = list(str2eval.keys())
            evaluations = ['Eval_EdgeOverlappingWithGold']
            try:
                for eval in evaluations:
                    df_global_matched = df_global_matched.append(str2eval[eval](network1=networks[0], network2=networks[1], method=method).evaluate(), ignore_index=True)
            except:
                print("Error in {}".format(method))
    print(df_global_matched)
    