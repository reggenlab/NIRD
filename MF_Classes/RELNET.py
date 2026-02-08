
############################################################
# Relevance network (RELNET) 
############################################################

# 1. Preparation of data frame for selected regulators
# 2. Initialize MI matrix and Compute MI for all pairs of regulators
# 3. Construct adjacency matrix by thresholding MI values

import numpy as np
import pandas as pd

class RELNET(object):

    def __init__(self, data, threshold=0.1, bins=10):
        """Initializes the RELNET class with data, threshold, and number of bins."""
        self.data = data
        self.threshold = threshold
        self.bins = bins
        self.regulators = self.get_regulators()

    def get_regulators(self):
        """Retrieves the regulators from the data, using '_tf_names' if provided, otherwise all features."""
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
        """Computes the mutual information matrix and applies the RELNET algorithm to generate the adjacency matrix."""
        # Preparing the data frame for selected regulators.
        inDF = pd.DataFrame(data=self.data['_data'], columns=self.data['_features'])
        inDF = inDF.loc[:, self.regulators]
        n_genes = len(self.regulators)

        # Computes the mutual information matrix for all pairs of regulators.
        mi_matrix = np.zeros((n_genes, n_genes))
        for i in range(n_genes):
            for j in range(i + 1, n_genes):
                mi_matrix[i, j] = mi_matrix[j, i] = self.mutual_information(inDF.iloc[:, i], inDF.iloc[:, j])

        # Constructs the adjacency matrix by thresholding the mutual information values.
        adjacency_matrix = np.where(mi_matrix > self.threshold, 1, 0)
        return adjacency_matrix, self.regulators