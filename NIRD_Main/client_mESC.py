# %% Imports

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import time
import argparse

from MF_Evaluation import str2eval, Eval_EdgeOverlappingWithGold, Eval_EdgeOverlapping
from MF_Datasets import str2dataset
from MF_Classes import str2method

from MF_Clients.helper import data2network   #viz


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run matrix factorization methods on biological datasets.')

    parser.add_argument(
        '--datasets',
        type=str,
        default='mESC',
        help="Comma-separated list of dataset names (e.g., 'mESC,human_knee')"
    )

    parser.add_argument(
        '--methods',
        type=str,
        default='PCA,SVD,NMF',
        help="Comma-separated list of method names (e.g., 'PCA,NMF,GENIE3')"
    )

    parser.add_argument(
        '--evaluations',
        type=str,
        default='Eval_EdgeOverlapping',
        help="Comma-separated list of evaluation function names (e.g., 'Eval_EdgeOverlapping,Eval_EdgeOverlappingWithGold')"
    )

    return parser.parse_args()


# %% Main

# if __name__ == '__main__':

#     df_global_matched = pd.DataFrame(columns=['xt', 'matches', 'method'])
#     # datasets = list(str2dataset.keys())
#     datasets = ['mESC']
#     for dataset in datasets:
#         data = str2dataset[dataset](dataset_name=dataset).get_dataset()
#         print(dataset)

#         # methods = list(str2method.keys())[::2][:2]
#         methods = ['PCA', 'SVD', 'NMF', 'ICM', 'BD', 'BMF', 'LSNMF', 'KLD_NMF', 'ENMF', 'PMF', 'SNMF', 'PMFCC', 'SepNMF', 'NIRD', 'ARACNE', 'RELNET', 'MRNET', 'C3NET', 'GENIE3', 'GrnBoost2',]
#         # methods = ['GENIE3']
#         print(methods)
#         for method in methods:
#             start_time = time.time()  # Start time
#             networks = [data2network(data=d, method=method) for d in data]

#             ######### Code for generating gene interaction matrix #############

#             # Extract the _corr and _regulators for gene interaction matrix
#             _corr = networks[0]['_corr']  # Correlation matrix
#             _regulators = networks[0]['_regulators']  # List of regulators (genes)

#             # Convert string representations to Python objects if necessary
#             if isinstance(_corr, str):
#                 _corr = ast.literal_eval(_corr)
#             if isinstance(_regulators, str):
#                 _regulators = ast.literal_eval(_regulators)

#             # Create a DataFrame for the gene interaction matrix
#             interaction_matrix = pd.DataFrame(_corr, index=_regulators, columns=_regulators)

#             # Optionally, save the gene interaction matrix to a CSV
#             interaction_matrix.to_csv(f'gene_interaction_matrix_{dataset}_{method}.csv')
#             print(f'Gene interaction matrix for {dataset} using {method} saved.')

#             ####################

#             # evaluations = list(str2eval.keys())[1:2]
#             evaluations = ['Eval_EdgeOverlapping']
#             try:
#                 for eval in evaluations:
#                     df_global_matched = df_global_matched.append(str2eval[eval](network1=networks[0], network2=networks[1], method=method).evaluate(), ignore_index=True)
#             except Exception as e:
#                 print("{}: {}".format(method, e))

#             end_time = time.time()  # End time
#             elapsed_time = end_time - start_time
#             print(f"Completed {method} in {elapsed_time:.2f} seconds.")
            # print("Completed...")
            
            # print(f"Compeled with global data shape as : {df_global_matched.shape}")
            # viz(df_global_matched)

    # Save the global dataframe to a CSV file
    # df_global_matched.to_csv('mESC_global_matched.csv', index=False)
    



if __name__ == '__main__':
    args = parse_arguments()

    datasets = [d.strip() for d in args.datasets.split(',')]
    methods = [m.strip() for m in args.methods.split(',')]
    evaluations = [e.strip() for e in args.evaluations.split(',')]

    df_global_matched = pd.DataFrame(columns=['xt', 'matches', 'method'])

    for dataset in datasets:
        data = str2dataset[dataset](dataset_name=dataset).get_dataset()
        print(f"Dataset: {dataset}")
        print(f"Methods: {methods}")

        for method in methods:
            start_time = time.time()

            networks = [data2network(data=d, method=method) for d in data]

            # Extract and save gene interaction matrix
            _corr = networks[0]['_corr']
            _regulators = networks[0]['_regulators']

            if isinstance(_corr, str):
                _corr = ast.literal_eval(_corr)
            if isinstance(_regulators, str):
                _regulators = ast.literal_eval(_regulators)

            interaction_matrix = pd.DataFrame(_corr, index=_regulators, columns=_regulators)
            interaction_matrix.to_csv(f'gene_intr_mat_{dataset}_{method}.csv')
            print(f'Gene interaction matrix for {dataset} using {method} saved.')

            try:
                for eval in evaluations:
                    df_global_matched = df_global_matched.append(
                        str2eval[eval](network1=networks[0], network2=networks[1], method=method).evaluate(),
                        ignore_index=True
                    )
            except Exception as e:
                print(f"{method}: {e}")

            elapsed_time = time.time() - start_time
            print(f"Completed {method} in {elapsed_time:.2f} seconds.")





# Code for generating combined plot of overlapping curves -------------------------

# if __name__ == '__main__':

#     df_global_matched = pd.DataFrame(columns=['xt', 'matches', 'method'])
#     datasets = ['mESC']
#     for dataset in datasets:
#         data = str2dataset[dataset](dataset_name=dataset).get_dataset()
#         print(dataset)

#         # methods = ['PCA', 'SVD', 'KLD_NMF', 'Kernel_PCA', 'NMF', 'BMF', 'SepNMF', 'ICM', 'ENMF', 'SNMF']
#         methods = ['PCA', 'SVD', 'SNMF', 'GENIE3', 'GrnBoost2', 'RELNET']
#         print(methods)
#         evaluator = Eval_EdgeOverlapping(network1=None, network2=None)
        
#         for method in methods:
#             networks = [data2network(data=d, method=method) for d in data]

#             evaluations = ['Eval_EdgeOverlapping']
#             try:
#                 for eval in evaluations:
#                     evaluator._set(network1=networks[0], network2=networks[1])
#                     evaluator.method = method  # Set the method name for the evaluator
#                     df_global_matched = evaluator.evaluate(plot=False)
#             except Exception as e:
#                 print("{}: {}".format(method, e))
#             print("Completed...")

#         # Plot all methods together after looping through all methods
#         evaluator.plot_matching_edges(f_name='../ankur/New_Figures/mESC/batches/competitor_algorithms.svg')