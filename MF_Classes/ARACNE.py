
#################################################################################################################
# Algorithm for the Reconstruction of Accurate Cellular Networks (ARACNE)
#################################################################################################################

# 1. Preparation of data frame for selected regulators
# 2. Compute mutual information for all pairs of regulators
# 3. Data Processing Inequality (DPI), for elimination of indirect interactions
# 4. Threshold MI values and construct adjacency matrix

import numpy as np
import pandas as pd

class ARACNE(object):

    def __init__(self, data, threshold=0.1, bins=10):  # apply threshold to remove weak interactions 
        self.data = data
        self.threshold = threshold
        self.bins = bins
        self.regulators = self.get_regulators()

    def get_regulators(self):
        if self.data['_tf_names'] is None:
            regulators = list(set(self.data['_features']))
        else:
            regulators = list(set(self.data['_features']) & set(self.data['_tf_names']))
        return regulators

    def mutual_information(self, x, y):
        """Calculates mutual information between two variables using a 2D histogram to estimate joint and marginal distributions."""
        joint_prob, _, _ = np.histogram2d(x, y, bins=self.bins)
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
        # Preparing the data frame for selected regulators.
        inDF = pd.DataFrame(data=self.data['_data'], columns=self.data['_features'])
        inDF = inDF.loc[:, self.regulators]
        n_genes = len(self.regulators)

        # Computes the mutual information matrix for all pairs of regulators.
        mi_matrix = np.zeros((n_genes, n_genes))
        for i in range(n_genes):
            for j in range(i + 1, n_genes):
                mi_matrix[i, j] = mi_matrix[j, i] = self.mutual_information(inDF.iloc[:, i], inDF.iloc[:, j])

        # Data Processing Inequality (DPI) to eliminate indirect interactions.
        for i in range(n_genes):
            for j in range(i + 1, n_genes):
                for k in range(n_genes):
                    if i != k and j != k:
                        if mi_matrix[i, k] > 0 and mi_matrix[j, k] > 0:
                            if mi_matrix[i, k] + mi_matrix[j, k] >= mi_matrix[i, j]:
                                mi_matrix[i, j] = 0
        
        # Constructs the adjacency matrix by thresholding the mutual information values.
        adjacency_matrix = np.where(mi_matrix > self.threshold, 1, 0)
        return adjacency_matrix, self.regulators

