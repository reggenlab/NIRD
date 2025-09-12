# %% imports
import numpy as np
import pandas as pd
import networkx as nx
from pygsp import graphs, filters


# %% methods

def get_top_vals(adj, top):
    adj = np.triu(adj)
    adj_return = np.zeros_like(adj)
    top_indices = np.unravel_index(np.argsort(adj, axis=None)[::-1][0:top], adj.shape)
    adj_return[top_indices] = adj[top_indices]
    adj_return = np.triu(adj_return) + np.transpose(np.triu(adj_return))
    return adj_return


def get_sorted_edges(G):
    edges_list = G.get_edge_list()
    edges = np.array(edges_list[0:2], dtype=np.int32)
    edges = edges.transpose()
    edges = edges[np.argsort(edges_list[2])[::-1], :]
    return edges

# remove loops
# set strongest edge for all feature pair
def prepare_corr(data):
    a = np.abs(data).copy()
    for i in range(a.shape[0]):
        for j in range(i, a.shape[1]):
            if i == j:
                a[i, j] = 0
            else:
                a[i, j] = max(a[i, j], a[j, i])
                a[j, i] = a[i, j]
    return a


def data2corr(data, method='pearson'):
    if method == 'pearson':
        adj = np.mat(prepare_corr(pd.DataFrame(data).corr(method=method).values))
    elif method == 'spearman':
        adj = np.mat(prepare_corr(pd.DataFrame(data).corr(method=method).values))

    return {'_method': method, '_adj': adj}


def remove_directions(adj):
    adj = np.nan_to_num(adj)
    adj = np.abs(adj)  # no negative weights
    adj = adj - np.diag(np.diag(adj))  # no self loops
    adj = np.triu(adj) + np.transpose(np.triu(adj))  # no directions
    return adj

def create_graph(adj, edges):
    adj = remove_directions(adj)
    top_vals = edges
    adj = get_top_vals(adj, top_vals)
    #     adj_g = np.abs(adj)
    G = graphs.Graph(adj)
    # print ("GSP object created : {0} nodes, {1} edges".format(G.N, G.Ne))
    return G

def adjlist2corr(adj, n_regulator):
    # print(adj)
    mat = np.zeros([n_regulator, n_regulator])
    for row in adj:
        mat[int(row[0][1:])-1, int(row[1][1:])-1] = float(row[2])
    # print(mat)
    return mat


# with pyGSP
# def corr2adjlist(corr, top=None):
#
#     if corr['_label'] == 'gold':
#         temp = corr['_corr'][corr['_corr'][:, 2] == 1]
#         TF = [int(c[1:]) for c in temp[:, 0]]
#         target = [int(c[1:]) for c in temp[:, 1]]
#         adj_list = list(zip(TF, target))
#
#     else:
#         if corr['_corr'].shape[1] == 3: # GENIE3 or GRNBoost
#             corr['_corr'] = adjlist2corr(adj=corr['_corr'], n_regulator=corr['_count'])
#         # networkX or pyGSP may be used
#         adj_list = get_sorted_edges(create_graph(adj=corr['_corr'], edges=top))
#
#     return adj_list

# def corr2adjlist(corr, top=None):
#
#     if corr['_corr'].shape[1] == 3: # gold or GENIE3 or GRNBoost
#         TF = [int(c[1:]) for c in corr['_corr'][:, 0]]
#         target = [int(c[1:]) for c in corr['_corr'][:, 1]]
#         adj_list = pd.DataFrame(data=corr['_corr'], columns=['TF', 'target', 'importance'])
#         adj_list.TF = TF
#         adj_list.target = target
#     else:
#         adj_list = pd.DataFrame([{'TF': i+1, 'target': j+1, 'importance': corr['_corr'][i, j]}
#                                  for i in range(corr['_corr'].shape[0])
#                                  for j in range(corr['_corr'].shape[1])])
#
#     if corr['_label'] == 'gold':
#         adj_list = adj_list[adj_list.importance != 0]
#     else:
#         adj_list = adj_list.sort_values(by='importance', ascending=False).head(top)
#
#     return adj_list[['TF', 'target']].values


def get_top_edges(matrix, top_n):
    flat = matrix.flatten()
    indices = np.argpartition(flat, -top_n)[-top_n:]
    row_indices, col_indices = np.unravel_index(indices, matrix.shape)
    top_values = flat[indices]
    top_edges = list(zip(row_indices, col_indices, top_values))
    top_edges.sort(key=lambda x: x[2], reverse=True)
    return top_edges

# own
def corr2adjlist(corr, top=None):
    if corr['_corr'].shape[1] == 3: # gold
        TF = [int(c[1:]) for c in corr['_corr'][:, 0]]
        target = [int(c[1:]) for c in corr['_corr'][:, 1]]
        adj_list = pd.DataFrame(data=corr['_corr'], columns=['TF', 'target', 'importance'])
        adj_list.TF = TF
        adj_list.target = target
        adj_list = adj_list.loc[adj_list['importance']!=0]

    # if corr['_label'] == 'gold':
    #     temp = corr['_corr'][corr['_corr'][:, 2] == 1]
    #     TF = [int(c[1:]) for c in temp[:, 0]]
    #     target = [int(c[1:]) for c in temp[:, 1]]
    #     adj_list = list(zip(TF, target))
    else:
        adj = corr['_corr']
        adj = adj - np.diag(np.diag(adj))
        adj_list = pd.DataFrame([{'TF': i + 1, 'target': j + 1, 'importance': adj[i, j]}
                                         for i in range(adj.shape[0])
                                         for j in range(adj.shape[1])])
        adj_list = adj_list.sort_values(by='importance', ascending=False).head(top)

    return adj_list[['TF', 'target']].values
