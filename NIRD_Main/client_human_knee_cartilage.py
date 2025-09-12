import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import time

from MF_Evaluation import str2eval, Eval_EdgeOverlapping
from MF_Datasets import str2dataset
from MF_Classes import str2method

from MF_Clients.helper import data2network # viz

# %% Main

if __name__ == '__main__':

    df_global_matched = pd.DataFrame(columns=['xt', 'matches', 'method'])

    # Specify the dataset list (human knee cartilage in this case)
    datasets = ['human_knee_cartilage']

    # Loop over the dataset(s)
    for dataset in datasets:
        data = str2dataset[dataset](dataset_name=dataset).get_dataset()  # Load human knee cartilage dataset
        print(dataset)

        # Specify the method(s) for processing (you can adjust or add more if needed)
        methods = ["GrnBoost2", "RELNET", "ARACNE"]
        # methods = ['PCA', 'SVD', 'NMF', 'ICM', 'BMF', 'BD', 'LSNMF', 'KLD_NMF', 'ENMF', 'PMF', 'SNMF', 'PMFCC', 'SepNMF', 'Kernel_PCA', 'ARACNE', 'RELNET', 'MRNET', 'C3NET', 'GENIE3', 'GrnBoost2']
        print(methods)

        # Loop over the methods
        for method in methods:
            start_time = time.time()  # Start time
            # Convert the datasets to networks based on the method
            networks = [data2network(data=d, method=method) for d in data]

            # ########## Code for generating gene interaction matrix #############

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
            # interaction_matrix.to_csv(f'gene_intr_mat_{dataset}_{method}.csv')
            # print(f'Gene interaction matrix for {dataset} using {method} saved.')

            # #####################

            # Specify the evaluation method(s)
            evaluations = ['Eval_Knee_Cartilage']

            try:
                # Loop over evaluations and compute the overlaps
                for eval in evaluations:
                    result = str2eval[eval](network1=networks[0], network2=networks[1], method=method).evaluate()
                    df_global_matched = df_global_matched.append(result, ignore_index=True)
            except Exception as e:
                print(f"Error with {method}: {e}")

            end_time = time.time()  # End time
            elapsed_time = end_time - start_time
            print(f"Completed {method} in {elapsed_time:.2f} seconds.")

            # print("Completed...")

#     # Save the global dataframe to a CSV file
#     #df_global_matched.to_csv('MF_Datasets/human_knee_cartilage/knee_cartilage_inferred_data.csv', index=False)





########################## Script for generating multiple overlapping curves in the same plot #################################

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# from MF_Evaluation import str2eval
# from MF_Datasets import str2dataset
# from MF_Clients.helper import data2network

# if __name__ == '__main__':
#     df_global_matched = pd.DataFrame()  # To accumulate results from all methods

#     # Specify the dataset list (human knee cartilage in this case)
#     datasets = ['human_knee_cartilage']

#     # Loop over the dataset(s)
#     for dataset in datasets:
#         data = str2dataset[dataset](dataset_name=dataset).get_dataset()  # Load human knee cartilage dataset
#         print(f"Processing dataset: {dataset}")

#         # Specify the method(s) for processing
#         # methods = ['NIRD', 'GENIE3', 'GrnBoost2', 'ARACNE', 'RELNET', 'MRNET', 'C3NET']
#         methods = ['PCA', 'NMF', 'BMF', 'LSNMF', 'KLD_NMF', 'ENMF', 'SNMF', 'PMFCC', 'SepNMF', 'NIRD']
#         print(f"Methods: {methods}")

#         # Loop over the methods
#         for method in methods:
#             # Convert the datasets to networks based on the method
#             networks = [data2network(data=d, method=method) for d in data]

#             # Specify the evaluation method(s)
#             evaluations = ['Eval_EdgeOverlapping']

#             try:
#                 # Loop over evaluations and compute the overlaps
#                 for eval in evaluations:
#                     result = str2eval[eval](network1=networks[0], network2=networks[1], method=method).evaluate(plot=False)
#                     df_global_matched = df_global_matched.append(result, ignore_index=True)
#             except Exception as e:
#                 print(f"Error with {method}: {e}")

#             print(f"Completed evaluation for method: {method}")

#     # Save the global dataframe to a CSV file
#     output_csv = '../ankur/New_Figures/Human_Knee_Cartilage/human_knee_cartilage_inferred_data.csv'
#     df_global_matched.to_csv(output_csv, index=False)
#     print(f"Results saved to {output_csv}")

#     # Plot all methods together after the loop
#     if not df_global_matched.empty:
#         plt.figure(figsize=(10, 6))

#         for method in df_global_matched['method'].unique():
#             df_method = df_global_matched[df_global_matched['method'] == method]
#             df_method['total_matches'] = np.cumsum(df_method['matches'])
#             df_method['y'] = (df_method['total_matches'] / (max(df_method['total_matches']) + 0.0001)) * 100

#             sns.lineplot(data=df_method, x="xt", y="y", label=method)

#         plt.xlabel(f"Edges in {networks[0]['_label']}")
#         plt.ylabel(f"Matches in {networks[1]['_label']} (in %)")
#         plt.title(f"Edge overlapping between {networks[0]['_label']} and {networks[1]['_label']} for different methods")
#         plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
#         plt.xlim(0, 10000)
#         plt.tight_layout()
#         output_plot = '../ankur/New_Figures/Human_Knee_Cartilage/MF_algorithms_overlapping.svg'
#         plt.savefig(output_plot)
#         print(f"Plot saved to {output_plot}")
#         plt.show()
