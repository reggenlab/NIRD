# %% Imports
import pandas as pd
import numpy as np

from sklearn.manifold import TSNE
from numpy import reshape
import matplotlib.pyplot as plt
import seaborn as sns

import MF_Classes

import warnings
warnings.filterwarnings('ignore')

# %% Methods

def csv2dict(data_filepath, tf_filepath=None, name=None):
	data = csv2df(data_filepath)
	tfs = csv2array(tf_filepath)
	dict = {'_name': name,
	        '_data': np.mat(data.values),
			'_index': data.index.values,
			'_features': data.columns.values,
			'_feat_count': len(data.columns),
			'_tf_names': tfs,
	        '_cluster': np.full(shape=(len(data.index.values),1), fill_value=name),
	        }
	return dict


def csv2array(tf_filepath):
	if tf_filepath is None:
		return None
	else:
		with open(tf_filepath) as file:
			return np.array(line.strip() for line in file.readlines())


def csv2df(data_filepath):
	return pd.read_csv(filepath_or_buffer=data_filepath,index_col=0, sep='\t').T


def tSNE_projection(data, viz=True):
	fig, ax = plt.subplots(2, 5, sharex=True, figsize=(16, 10))
	for p1 in range(5, 55, 5): # maybe done using pool.map
		i = int((p1 - 5) / 5)
		a = int(i / 5)
		b = i % 5

		print('Data: <{}> || Perplexity: <{}>'.format(data['_name'], p1))
		tsne = TSNE(n_components=2, verbose=0, random_state=123, n_jobs=-1, perplexity=p1)
		z = tsne.fit_transform(data['_data'])

		df = pd.DataFrame()
		df["comp-1"] = z[:, 0]
		df["comp-2"] = z[:, 1]
		df["y"] = data['_cluster']

		if viz:
			sns.set_palette(sns.color_palette("hls", 2))
			sns.scatterplot(
				x="comp-1",
				y="comp-2",
				hue=df["y"].to_list(),
				data=df,
				ax=ax[a,b]
			).set(title="T cells[{}] \nwith perplexity={}".format(data['_name'], p1))

			viz_file_path = '../ankur/New_Figures/T_HelperCell'
			# viz_file_path = '../MF_Figures/t_HelperCell'
			viz_name = viz_file_path + '/' + data['_name'] + "_" + 'tsne' + '.svg'
	sns.set(title="tSNE Projection for t-cells")
	plt.savefig(viz_name)
	plt.show()


def merged(d1, d2):
	dict = {'_name': 't-cells',
	        '_data': np.concatenate((d1['_data'], d2['_data']), axis=0),
	        '_index': np.concatenate((d1['_index'],d2['_index']), axis=0),
	        '_features': np.concatenate((d1['_features'],d2['_features']), axis=0),
	        '_feat_count': d1['_feat_count'],
	        '_tf_names': d1['_tf_names'], # to check
	        '_cluster': np.concatenate((d1['_cluster'],d2['_cluster']), axis=0),
	        }
	return dict



# %% Main

if __name__ == '__main__':
	naive_data_filepath = '../Immune_data_T_helper_cell/filtered_gene_bc_matrices_naive/10X/final_mat.txt'
	th2_data_filepath = '../Immune_data_T_helper_cell/filtered_gene_bc_matrices_th2/10X/final_mat.txt'

	naive = csv2dict(data_filepath=naive_data_filepath, name='naive')
	th2 = csv2dict(data_filepath=th2_data_filepath, name='th2')

	# Viz using tSNE
	tSNE = False
	if tSNE:
		tSNE_projection(data=naive)
		tSNE_projection(data=th2)
		tSNE_projection(data=merged(naive, th2))

	# Network Inference
	network_inference = True
	if network_inference:
		m1 = MF_Classes.SingularValueDecomposition(data=naive)
		network1 = m1.fit()

		m2 = MF_Classes.SingularValueDecomposition(data=th2)
		network2 = m2.fit()

	print('Hi')

