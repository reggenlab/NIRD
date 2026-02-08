
#######################################################################
# Conservative Causal Core NETwork (C3NET)
#######################################################################

# 1. Preparation of data frame for selected regulators
# 2. Compute mutual information for all pairs of regulators
# 3. Eliminate non-significant connections (Significance Filtering)
# 4. Select most significant edge for each gene

import numpy as np
import pandas as pd
from scipy.stats import chi2

class C3NET(object):

    def __init__(self, data, alpha=0.01):
        self.data = data
        self.alpha = alpha
        self.regulators = self.get_regulators()

    def get_regulators(self):
        if self.data['_tf_names'] is None:
            regulators = list(set(self.data['_features']))
        else:
            regulators = list(set(self.data['_features']) & set(self.data['_tf_names']))
        return regulators

    def mutual_information(self, x, y):
        joint_prob, _, _ = np.histogram2d(x, y, bins=10)
        joint_prob /= len(x)

        prob_x = np.sum(joint_prob, axis=1)
        prob_y = np.sum(joint_prob, axis=0)

        mi = 0
        for i in range(len(prob_x)):
            for j in range(len(prob_y)):
                if joint_prob[i, j] > 0 and prob_x[i] > 0 and prob_y[j] > 0:
                    mi += joint_prob[i, j] * np.log2(joint_prob[i, j] / (prob_x[i] * prob_y[j]))

        return mi

    def fit(self):
        # Preparing the data frame for selected regulators
        inDF = pd.DataFrame(data=self.data['_data'], columns=self.data['_features'])
        inDF = inDF.loc[:, self.regulators]
        n_genes = len(self.regulators)

        # Compute pairwise mutual information for all pairs of regulators
        mi_matrix = np.zeros((n_genes, n_genes))
        for i in range(n_genes):
            for j in range(i + 1, n_genes):
                mi_matrix[i, j] = mi_matrix[j, i] = self.mutual_information(inDF.iloc[:, i], inDF.iloc[:, j])

        # Apply significance filtering using the chi-squared test
        p_values = 1 - chi2.cdf(mi_matrix, df=1)
        significant_matrix = np.where(p_values < self.alpha, mi_matrix, 0)

        # Select the most significant edge for each gene
        adjacency_matrix = np.zeros_like(significant_matrix)
        for i in range(n_genes):
            max_index = np.argmax(significant_matrix[i, :])
            if significant_matrix[i, max_index] > 0:
                adjacency_matrix[i, max_index] = 1
                adjacency_matrix[max_index, i] = 1  # Make it symmetric

        return adjacency_matrix, self.regulators
