import numpy as np
from MF_Classes import str2method
import seaborn as sns
import matplotlib.pyplot as plt


def do_feature_selection(df):
	q50 = df.quantile(axis=0, q=.50)  # where >50% values are non-zero
	q05 = df.quantile(axis=0, q=.05)  # Identification of housekeeping genes

	to_keep = np.where(q50 > 0)[0]
	to_drop = np.where(q05 > 0)[0]
	to_keep = list(set(to_keep) - set(to_drop))

	df = df.iloc[:, to_keep]
	return df


def data2network(data, method):
	if data['_label'] == 'gold':
		net_dict = {'_label': data['_label'], '_corr': data['_edges']}

	else:
		net, regulators = str2method[method](data=data).fit()
		# print(net)
		net_dict = {'_label': data['_label'],
		            '_count': len(data['_tf_names']) if data['_tf_names'] is not None else data['_feat_count'],
		            '_p_label': data['_p_label'],
		            '_corr': net,
		            '_regulators': regulators}
	return net_dict


# def viz(df, method, f_name='plot.png'):
# 	df['total_matches'] = np.cumsum(df.matches)
# 	df['y'] = (df['total_matches'] / 10000) * 100

# 	# sns.set(style="whitegrid")
# 	plt.figure(figsize=(10, 6))

# 	method_name = method + " based : "
# 	p = sns.lineplot(data=df, x="xt", y="y")
# 	p.set(xlabel="edges in smartSeq",
# 		  ylabel="matches in dropSeq (in %)",
# 		  title=method_name + "Edge overlapping " + self.network1['_label'] + " and " + self.network2['_label'])
# 	plt.savefig(f_name)
# 	plt.show()
