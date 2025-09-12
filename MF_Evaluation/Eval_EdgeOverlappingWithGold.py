import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import integrate

from .MF_Evaluator import MF_Evaluator
from MF_Evaluation.helper import corr2adjlist

import warnings
warnings.filterwarnings('ignore')


class Eval_EdgeOverlappingWithGold(MF_Evaluator):

    def __init__(self, network1, network2, method=None, top=50000, dx=50, outdir="./") -> None:
        super().__init__(network1, network2, method)

        self.df_global = pd.DataFrame(columns=['xt', 'matches', 'method'])
        self.top = top
        self.dx = dx
        self.p_label = network1['_p_label']
        self.outdir = outdir

        # Create output directories for plots if they don't exist
        os.makedirs(self.outdir, exist_ok=True)

    def _set(self, network1, network2=None):
        self.network1 = network1
        self.network2 = network2

    def evaluate(self):
        # Dynamic filename for plot
        plot_dir = os.path.join(
        self.outdir
        )
        os.makedirs(plot_dir, exist_ok=True)

        plot_path = os.path.join(plot_dir, f"{self.method}.svg")
        matches = self.get_matching_edges(plot=True, f_name=plot_path)
        print(f"Evaluation completed for {self.method}. Plot saved: {plot_path}")
        return matches

    def get_matching_edges(self, plot=True, f_name='plot.svg'):
        df_temp = pd.DataFrame(columns=['xt', 'matches'])

        g1 = corr2adjlist(corr=self.network1, top=self.top)
        g2 = corr2adjlist(corr=self.network2)

        e2 = np.array(g2, dtype=np.int32).transpose().tolist()
        for xt in range(0, g1.shape[0], self.dx):
            e1 = np.array(g1, dtype=np.int32)[xt:xt + self.dx, :].transpose().tolist()
            e1 = np.concatenate((e1, e1[::-1]), axis=1).tolist()

            df_temp = df_temp.append(
                {'xt': xt, 'matches': len(set(zip(*e1)) & set(zip(*e2)))},
                ignore_index=True
            )

        df_match = df_temp.copy()
        df_match['method'] = self.method
        self.df_global = self.df_global.append(df_match, ignore_index=True)

        if plot:
            self.plot_matching_edges(df_match, f_name)
        return df_temp

    def plot_matching_edges(self, df_match, f_name='plot.svg'):
        df_match['total_matches'] = np.cumsum(df_match.matches)
        df_match['y'] = (df_match['total_matches'] / (max(df_match['total_matches']) + 1e-6)) * 100

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_match, x="xt", y="y", label=self.method)
        plt.xlabel("Edges in " + self.network1['_label'])
        plt.ylabel("Matches in " + self.network2['_label'] + " (in %)")
        plt.title(f"Edge Overlapping: {self.network1['_label']} vs {self.network2['_label']}")
        plt.legend(title="Methods")
        plt.tight_layout()

        # Save plot
        plt.savefig(f_name)
        plt.close()

        # Calculate AUC
        auc = integrate.simps(x=df_match['xt'], y=df_match['total_matches'])
        max_auc = integrate.simps(
            x=list(df_match.xt),
            y=len(list(df_match.total_matches)) * [max(list(df_match.total_matches))]
        )
        auc_score = auc / max_auc
        print(f"AUC Score ({self.method}): {auc_score:.4f}")
