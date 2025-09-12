# %% Imports

import pandas as pd
from MF_Evaluation import str2eval
from MF_Datasets import str2dataset
from MF_Classes import str2method
import ast

from MF_Clients.helper import data2network

# %% Main


if __name__ == '__main__':

    datasets = ['mESC_gold']
    for dataset in datasets:
        data = str2dataset[dataset](dataset_name=dataset).get_dataset()
        print(dataset)

        # methods = list(str2method.keys())
        methods = ['PCA', 'SVD', 'NMF', 'ICM', 'BD', 'BMF', 'LSNMF', 'KLD_NMF', 'ENMF', 'PMF', 'SNMF', 'PMFCC', 'SepNMF', 'Kernel_PCA', 'GENIE3', 'GrnBoost2', 'ARACNE', 'RELNET', 'MRNET', 'C3NET']
        print(methods)
        
        for method in methods:
            for d in data:
                # x = data2network(data=d, method=method)
                networks = [data2network(data=d, method=method) for d in data]

                ######### Code for generating gene interaction matrix #############

            # Extract the _corr and _regulators for gene interaction matrix
            _corr = networks[0]['_corr']  # Correlation matrix
            _regulators = networks[0]['_regulators']  # List of regulators (genes)

            # Convert string representations to Python objects if necessary
            if isinstance(_corr, str):
                _corr = ast.literal_eval(_corr)
            if isinstance(_regulators, str):
                _regulators = ast.literal_eval(_regulators)

            # Create a DataFrame for the gene interaction matrix
            interaction_matrix = pd.DataFrame(_corr, index=_regulators, columns=_regulators)

            # Optionally, save the gene interaction matrix to a CSV
            interaction_matrix.to_csv(f'gene_interaction_matrix_{dataset}_{method}.csv')
            print(f'Gene interaction matrix for {dataset} using {method} saved.')

            #####################

            # evaluations = list(str2eval.keys())
            evaluations = ['Eval_EdgeOverlappingWithGold']
            try:
                for eval in evaluations:
                    str2eval[eval](network1=networks[0], network2=networks[1], method=method).evaluate()
            except Exception as e:
                print("{}: {}".format(method, e))
            # for eval in evaluations:
            #     str2eval[eval](network1=networks[0], network2=networks[1], method=method).evaluate()



# %% Imports

# from MF_Evaluation import str2eval
# from MF_Datasets import str2dataset
# from MF_Classes import str2method

# from MF_Clients.helper import data2network

# # %% Main


# if __name__ == '__main__':

#     datasets = ['mESC_gold']
#     for dataset in datasets:
#         data = str2dataset[dataset](dataset_name=dataset).get_dataset()
#         print(dataset)

#         # methods = list(str2method.keys())[::2]
#         # methods = ['GENIE3','GrnBoost2']
#         methods = ['PCA', 'SVD', 'NMF']
#         for method in methods:
#             for d in data:
#                 # x = data2network(data=d, method=method)
#                 networks = [data2network(data=d, method=method) for d in data]

#             # evaluations = list(str2eval.keys())
#             evaluations = ['Eval_EdgeOverlappingWithGold']
#             try:
#                 for eval in evaluations:
#                     str2eval[eval](network1=networks[0], network2=networks[1], method=method).evaluate()
#             except Exception as e:
#                 print("{}: {}".format(method, e))
#             # for eval in evaluations:
#             #     str2eval[eval](network1=networks[0], network2=networks[1], method=method).evaluate()
