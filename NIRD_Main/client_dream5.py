
# %% Imports
import pandas as pd
import numpy as np
import ast
import time

from MF_Evaluation import str2eval
from MF_Datasets import str2dataset
from MF_Classes import str2method

from MF_Clients.helper import data2network

# # %% Main

if __name__ == '__main__':

    df_global_matched = pd.DataFrame()
    global_net = list()
    features = list()

    datasets = ['dream5_net4']#, 'dream5_net2', 'dream5_net3', 'dream5_net4']
    for dataset in datasets:
        data = str2dataset[dataset](dataset_name=dataset).get_dataset()
        print(dataset)
        features = data[0]['_features']
        # methods = ['ARACNE']
        # methods = ['PCA', 'SVD', 'NMF', 'ICM', 'BD', 'BMF', 'LSNMF', 'KLD_NMF', 'ENMF', 'PMF', 'SNMF', 'PMFCC', 'SepNMF', 'NIRD']
        # methods = list(str2method.keys())[::2][:2]
        methods = ['PCA', 'SVD', 'NMF', 'ICM', 'BD', 'BMF', 'LSNMF', 'KLD_NMF', 'ENMF', 'PMF', 'SNMF', 'PMFCC', 'SepNMF', 'NIRD', 'ARACNE', 'RELNET', 'MRNET', 'C3NET', 'GENIE3', 'GrnBoost2']
        print(methods)
        for method in methods:
            start_time = time.time()  # Start time
            networks = [data2network(data=d, method=method) for d in data]
            global_net.append(networks[0]['_corr'])

            ########## Code for generating gene interaction matrix #############

            # # Extract the _corr and _regulators for gene interaction matrix
            # _corr = networks[0]['_corr']  # Correlation matrix
            # _regulators = networks[0]['_regulators']  # List of regulators (genes)

            # # Convert string representations to Python objects if necessary
            # if isinstance(_corr, str):
            #     _corr = ast.literal_eval(_corr)
            # if isinstance(_regulators, str):
            #     _regulators = ast.literal_eval(_regulators)

            # # Create a DataFrame for the gene interaction matrix
            # interaction_matrix = pd.DataFrame(_corr, index=_regulators, columns=_regulators)

            # # Optionally, save the gene interaction matrix to a CSV
            # interaction_matrix.to_csv(f'gene_interaction_matrix_{dataset}_{method}.csv')
            # print(f'Gene interaction matrix for {dataset} using {method} saved.')

            #####################

            # evaluations = list(str2eval.keys())
            evaluations = ['Eval_EdgeOverlappingWithGold']
            try:
                for eval in evaluations:
                    df_global_matched = df_global_matched.append(str2eval[eval](network1=networks[0], network2=networks[1], method=method).evaluate(), ignore_index=True)
            except:
                print("Error in {}".format(method))

            end_time = time.time()  # End time
            elapsed_time = end_time - start_time
            print(f"Completed {method} in {elapsed_time:.2f} seconds.")
            # print("Completed ...")
    
    # Save the global dataframe to a CSV file
    # df_global_matched.to_csv('../ankur/New_Figures/dream5/net1/gold/results.csv', index=False)

    




