
################################################################
# Minimum Redundancy networks(MRNET)
################################################################

# 1. Preparation of data frame for selected regulators
# 2. Initialize MI matrix and Compute MI for all pairs of regulators
# 3. Feature Selection : select top connections (genes that have highest MI value)
# 4. Network Construction

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

class MRNET:
    def __init__(self, data, num_connections=3):
        self.data = data
        self.num_connections = num_connections
        self.regulators = self.get_regulators()
        self.features = data['_features']

    def get_regulators(self):
        if self.data['_tf_names'] is None:
            return list(set(self.data['_features']))
        else:
            return list(set(self.data['_features']) & set(self.data['_tf_names']))

    def mutual_information(self, x, y):
        return mutual_info_score(x, y)

    def fit(self):
        # Preparing the data for selected regulators
        inDF = pd.DataFrame(data=self.data['_data'], columns=self.features)
        inDF = inDF.loc[:, self.regulators]
        n_genes = len(self.regulators)

        # Initialize mutual information matrix
        mi_matrix = np.zeros((n_genes, n_genes))

        # Calculate mutual information for all pairs of genes
        for i in range(n_genes):
            for j in range(i + 1, n_genes):
                mi = self.mutual_information(inDF.iloc[:, i], inDF.iloc[:, j])
                mi_matrix[i, j] = mi_matrix[j, i] = mi

        # Apply MRNET procedure to select top connections
        adjacency_matrix = np.zeros_like(mi_matrix)
        for i in range(n_genes):
            remaining_genes = list(range(n_genes))
            remaining_genes.remove(i)

            # Select top mutual information connections for each gene
            top_connections = sorted(
                remaining_genes, key=lambda j: mi_matrix[i, j], reverse=True
            )[:self.num_connections]

            for j in top_connections:
                adjacency_matrix[i, j] = mi_matrix[i, j]

        return adjacency_matrix, self.regulators

