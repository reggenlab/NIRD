import numpy as np
import multiprocessing as mp
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import BaseDecisionTree

def compute_feature_importances(estimator):
    """Extracts feature importances from a trained model."""
    if isinstance(estimator, BaseDecisionTree):
        return estimator.feature_importances_
    elif hasattr(estimator, 'estimators_'):
        return np.mean([tree.feature_importances_ for tree in estimator.estimators_], axis=0)
    else:
        raise ValueError("Unsupported model type for feature importance extraction.")

def get_link_list(VIM, gene_names, regulators=None, maxcount=None, file_name=None):
    """Generates ranked gene interactions from the variable importance matrix."""
    num_genes = VIM.shape[0]
    interactions = []
    for i in range(num_genes):
        for j in range(num_genes):
            if regulators is None or gene_names[j] in regulators:
                interactions.append((gene_names[j], gene_names[i], VIM[i, j]))
    interactions.sort(key=lambda x: x[2], reverse=True)
    
    if maxcount:
        interactions = interactions[:maxcount]
    
    if file_name:
        with open(file_name, 'w') as f:
            for reg, target, score in interactions:
                f.write(f"{reg}\t{target}\t{score}\n")
    return interactions

def GRNBoost2_(expr_data, gene_names, regulators=None, tree_method='RF', K=None, ntrees=1000, nthreads=1):
    """GRNBoost2 implementation using tree-based models."""
    num_genes = expr_data.shape[1]
    VIM = np.zeros((num_genes, num_genes))
    input_idx = np.arange(num_genes)
    
    if regulators is not None:
        input_idx = np.array([i for i, g in enumerate(gene_names) if g in regulators])
    
    pool = mp.Pool(nthreads)
    results = pool.map(wr_GRNBoost2_single, [(expr_data, i, input_idx, tree_method, K, ntrees) for i in range(num_genes)])
    pool.close()
    pool.join()
    
    for i, res in enumerate(results):
        VIM[i, input_idx] = res
    return VIM

def GRNBoost2_single(expr_data, output_idx, input_idx, tree_method, K, ntrees):
    """Trains a tree model to predict expression of a target gene."""
    X = expr_data[:, input_idx]
    y = expr_data[:, output_idx]
    
    if K is None:
        max_features = 'sqrt'
    else:
        max_features = min(K, len(input_idx))
    
    if tree_method == 'RF':
        model = RandomForestRegressor(n_estimators=ntrees, max_features=max_features, n_jobs=1)
    else:
        model = ExtraTreesRegressor(n_estimators=ntrees, max_features=max_features, n_jobs=1)
    
    model.fit(X, y)
    return compute_feature_importances(model)

def wr_GRNBoost2_single(args):
    """Wrapper function for multiprocessing."""
    return GRNBoost2_single(*args)
