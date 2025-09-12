# %% Imports

from MF_Evaluation import str2eval
from MF_Datasets import str2dataset
from MF_Classes import str2method

from MF_Clients.helper import data2network

# %% Main


if __name__ == '__main__':

    # datasets = list(str2dataset.keys())
    datasets = ['beeline_grnvbem']
    for dataset in datasets:
        data = str2dataset[dataset](dataset_name=dataset).get_dataset()
        print(dataset)

        methods = list(str2method.keys())[::2]
        # methods = ['GENIE3', 'GrnBoost2']
        for method in methods:
            networks = [data2network(data=d, method=method) for d in data]

            evaluations = list(str2eval.keys())
            # evaluations = ['Eval_EdgeOverlapping']
            try:
                for eval in evaluations:
                    str2eval[eval](network1=networks[0], network2=networks[1], method=method).evaluate()
            except Exception as e:
                print("{}: {}".format(method, e))
            # for eval in evaluations:
            #     str2eval[eval](network1=networks[0], network2=networks[1], method=method).evaluate()
