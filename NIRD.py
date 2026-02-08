
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import time
import argparse

from MF_Evaluation import str2eval, Eval_EdgeOverlappingWithGold, Eval_EdgeOverlapping
from MF_Datasets import str2dataset
from MF_Classes import str2method
from MF_Clients.helper import data2network   # viz


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run matrix factorization methods on biological datasets.')

    # Dataset selection
    parser.add_argument(
        '--datasets',
        type=str,
        default='double_expr',
        choices=['single_expr', 'double_expr', 'gold_data'],
        help="Dataset name: single_expr, double_expr or gold_data"
    )

    # Methods
    parser.add_argument(
        '--methods',
        type=str,
        default='SVD',
        help="Comma-separated list of method names (e.g., PCA, SVD, NMF, ICM, BD, BMF, LSNMF, KLD_NMF, ENMF, PMF, SNMF, PMFCC, SepNMF, Kernel_PCA, ARACNE, RELNET, MRNET, C3NET, GENIE3, GrnBoost2)"
    )

    # Evaluations
    parser.add_argument(
        '--evaluations',
        type=str,
        default='Eval_EdgeOverlapping',
        help="Comma-separated list of evaluation function names (Eval_EdgeOverlapping, Eval_EdgeOverlappingWithGold)"
    )

    # Enable or disable evaluation
    parser.add_argument(
        '--do_eval',
        action='store_true',
        help="If set, perform evaluation and generate plots; otherwise, skip evaluation."
    )

    # Input files for single_expr & double_expr
    parser.add_argument('--file1', type=str, help="For single_expr: expression data file | For double_expr: first expression data file")
    parser.add_argument('--file2', type=str, help="For double_expr: second expression data file")

    # Input files for gold_data
    parser.add_argument('--expr_file', type=str, help="For gold_data: expression data file (.tsv)")
    parser.add_argument('--tf_file', type=str, help="For gold_data: transcription factors file")
    parser.add_argument('--gold_file', type=str, help="For gold_data: gold standard file")

    # Output directory
    parser.add_argument(
        '--outdir',
        type=str,
        required=True,
        help="Directory where inferred networks and results will be saved"
    )

    return parser.parse_args()


def print_banner():
    banner = r"""
    ███╗   ██╗██╗██████╗██████╗
    ████╗  ██║██║██╔══██╗██╔══██╗
    ██╔██╗ ██║██║██████╔╝██║  ██║
    ██║╚██╗██║██║██╔══██╗██║  ██║
    ██║ ╚████║██║██║  ██║██████╔╝
    ╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝╚═════╝

     NIRD : Network Inference By Reducing Dimensions
    ---------------------------------------------------------------------------------
       Maintained & Copyright © RegGen Lab, IIIT-D
       Developers: Indra Prakash Jha & Ankur Gajendra Meshram
       Instructor/Supervisor: Dr. Vibhor Kumar
    ---------------------------------------------------------------------------------
    """
    print(banner)


# %% Main
if __name__ == '__main__':
    args = parse_arguments()
    print_banner()

    # Create output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)

    datasets = [d.strip() for d in args.datasets.split(',')]
    methods = [m.strip() for m in args.methods.split(',')]
    evaluations = [e.strip() for e in args.evaluations.split(',')]

    df_global_matched = pd.DataFrame(columns=['xt', 'matches', 'method'])

    print("\nStarting Network Inference Process...")

    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")

        # -------------------------------
        # Handle single_expr dataset
        # -------------------------------
        if dataset == "single_expr":
            if not args.file1:
                raise ValueError("For single_expr, you must provide --file1.")
            
            data = str2dataset[dataset](
                dataset_name=dataset,
                file1=args.file1
            ).get_dataset()

        # -------------------------------
        # Handle double_expr dataset
        # -------------------------------
        elif dataset == "double_expr":
            if not args.file1 or not args.file2:
                raise ValueError("For double_expr, you must provide both --file1 and --file2.")
            
            data = str2dataset[dataset](
                dataset_name=dataset,
                file1=args.file1,
                file2=args.file2
            ).get_dataset()

        # -------------------------------
        # Handle gold_data dataset
        # -------------------------------
        elif dataset == "gold_data":
            if not args.expr_file or not args.tf_file or not args.gold_file:
                raise ValueError("For gold_data, you must provide --expr_file, --tf_file, and --gold_file.")
            
            data = str2dataset[dataset](
                dataset_name=dataset,
                expr_file=args.expr_file,
                tf_file=args.tf_file,
                gold_file=args.gold_file
            ).get_dataset()

        else:
            raise ValueError(f"Dataset '{dataset}' is not supported.")

        print(f"Methods to run: {methods}")

        for method in methods:
            start_time = time.time()
            print(f"\nRunning {method} on {dataset}...")

            # Convert dataset into networks
            networks = [data2network(data=d, method=method) for d in data]

            # Extract and save inferred network matrix
            _corr = networks[0]['_corr']
            _regulators = networks[0]['_regulators']

            if isinstance(_corr, str):
                _corr = ast.literal_eval(_corr)
            if isinstance(_regulators, str):
                _regulators = ast.literal_eval(_regulators)

            interaction_matrix = pd.DataFrame(_corr, index=_regulators, columns=_regulators)

            # Save inferred network in output directory
            output_file = os.path.join(args.outdir, f'nird_net_{dataset}_{method}.csv')
            interaction_matrix.to_csv(output_file)
            print(f'Inferred network saved to: {output_file}')

            # Perform evaluation if requested
            if args.do_eval:
                print(f"Running evaluation for {method}...")
                try:
                    for eval in evaluations:
                        evaluator = str2eval[eval](
                            network1=networks[0],
                            network2=networks[1],
                            method=method,
                            outdir=args.outdir
                        )
                        df_global_matched = pd.concat([df_global_matched, evaluator.evaluate()],
                                                      ignore_index=True)
                except Exception as e:
                    print(f"Error in evaluation for {method}: {e}")

            elapsed_time = time.time() - start_time
            print(f"Completed {method} in {elapsed_time:.2f} seconds.")

    print("\nProcess completed successfully!")


    # # Save global evaluation results
    # global_eval_file = os.path.join(args.outdir, "global_evaluation_results.csv")
    # df_global_matched.to_csv(global_eval_file, index=False)
    # print(f"\nAll evaluations completed. Results saved: {global_eval_file}")
    