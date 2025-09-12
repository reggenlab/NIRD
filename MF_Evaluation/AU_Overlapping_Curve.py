# %% imports
import sys, os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from optparse import OptionParser

from MF_Classes import str2method
from MF_Evaluation.helper import corr2adjlist

import warnings
warnings.filterwarnings('ignore')

# %% Methods


top = 10000
dx = 500
df_global = pd.DataFrame(columns=['xt', 'matches', 'method'])

def parse_args(args):
    parser = OptionParser()

    parser.add_option('', '--algo', type = 'str',
                      help='Algorithm to run. Can either by GENIE3 or GRNBoost2')

    parser.add_option('', '--inFile', type='str',
                      help='Path to input tab-separated expression SamplesxGenes file')

    parser.add_option('', '--outFile', type = 'str',
                      help='File where the output network is stored')

    (opts, args) = parser.parse_args(args)

    return opts, args


def get_matching_edges(corr1, corr2, top=top, dx=dx, plot=False, method='', f_name='plot.png', with_gold=False):
    global df_global
    df_match = pd.DataFrame(columns=['xt', 'matches'])

    g1 = corr1.copy() if with_gold else corr2adjlist(corr=corr1, top=top)
    g2 = corr2adjlist(corr=corr2)

    for xt in range(0, g2.shape[0], dx):
        e1 = np.array(g1[:, :2], dtype=np.int32).transpose().tolist()
        e2 = np.array(g2[:, :2], dtype=np.int32)[xt:xt + dx, :].transpose().tolist()

        df_match = df_match.append({'xt': xt, 'matches': (len(set(zip(*e1)) & set(zip(*e2))))}, ignore_index=True)

    df_temp = df_match.copy()
    df_temp['method'] = method
    df_global = df_global.append(df_temp, ignore_index=True)
    if plot:
        plot_matching_edges(df_match, method, f_name)
    return df_match


def get_matching_edges_with_goldset(corr, gold, top=top, dx=dx, plot=False, method='', f_name='plot.png'):
    global df_global
    df_match = pd.DataFrame(columns=['xt', 'matches'])
    g1 = corr2adjlist(corr=corr, top=top)
    g2 = gold.copy()

    for xt in range(0, g2.shape[0], dx):
        e1 = np.array(g1[:, :2], dtype=np.int32).transpose().tolist()
        e2 = np.array(g2[:, :2], dtype=np.int32)[xt:xt + dx, :].transpose().tolist()

        df_match = df_match.append({'xt': xt, 'matches': (len(set(zip(*e1)) & set(zip(*e2))))}, ignore_index=True)

    df_temp = df_match.copy()
    df_temp['method'] = method
    df_global = df_global.append(df_temp, ignore_index=True)
    if plot:
        plot_matching_edges(df_match, method, f_name)
    return df_match


def plot_matching_edges(df_match, method, f_name='plot.png'):
    df_match['total_matches'] = np.cumsum(df_match.matches)
    df_match['y'] = (df_match['total_matches'] / top) * 100
    method_name = method.strip().split('/')[-1] + " based : "
    p = sns.lineplot(data=df_match, x="xt", y="y")
    p.set(xlabel="edges in smartSeq", ylabel="matches in dropSeq (in %)",
          title=method_name+"Edge overlapping between smartSeq and dropSeq")
    plt.savefig(f_name)
    plt.show()


def plot_global(df, f_name='global.svg'):

    df['total_matches'] = np.array([np.cumsum(df[df['method']==me].matches).values for me in df['method'].unique()]).flatten()
    df['y'] = (df['total_matches'] / top) * 100

    p = sns.lineplot(data=df, x="xt", y="y", hue='method')
    p.set(xlabel="edges in smartSeq", ylabel="matches in dropSeq (in %)",
          title="Edge overlapping between smartSeq and dropSeq")
    plt.savefig(f_name)
    plt.show()

    # total_area = simps(np.full((1, len(df.xt)/2), max(df.y)))
    # df_auc = pd.DataFrame(zip(methods, [simps(df[df['method'] == me].y)/total_area for me in df['method'].unique()]), columns=['method', 'auc'])
    # p1 = sns.catplot(data=df_auc, x='method', y='auc', kind='bar')
    # # plt.savefig('auc.svg')
    # plt.show()


def compare_between(method, data1, data2, f_name):

    m1 = str2method[method](data=data1)
    m2 = str2method[method](data=data2)

    network1 = m1.fit()
    network2 = m2.fit()
    matches = get_matching_edges(corr1=network1, corr2=network2, plot=True, method=method, f_name=f_name, with_gold=False)


def compare_with_goldset(method, data, gold, f_name):
    network1 = pd.read_csv(gold, sep=',', header=None).values
    m = str2method[method](data=data)
    network2 = m.fit()
    print(network)

    matches = get_matching_edges(corr1=network1, corr2=network2, plot=True, method=method, f_name=f_name, with_gold=True)
    print(matches)


def Eval_Overlapping_Edges_Between_Networks(method, data1, data2, with_gold=False):

    if with_gold:
        compare_with_goldset(method=method, data=data1, gold=data2, f_name='../ankur/New_Figures/mESC/gold/'+method+'.svg')
    else:
        compare_between(method=method, data1=data1, data2=data2, f_name='../ankur/New_Figures/mESC/batches/'+method+'.svg')
