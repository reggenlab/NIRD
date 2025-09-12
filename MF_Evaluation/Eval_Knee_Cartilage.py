
# from .MF_Evaluator import MF_Evaluator
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy import integrate
# from MF_Evaluation.helper import corr2adjlist
# import warnings
# warnings.filterwarnings('ignore')


# class Eval_Knee_Cartilage(MF_Evaluator):
# 	def __init__(self, network1, network2, method=None, top=10000, dx=500) -> None:
# 		super().__init__(network1, network2, method)

# 		self.df_global = pd.DataFrame(columns=['xt', 'matches', 'method'])
# 		self.top = top
# 		self.dx = dx

# 	def _set(self, network1, network2=None):
# 		self.network1 = network1
# 		self.network2 = network2

# 	def evaluate(self):
# 		matches = self.get_matching_edges(plot=True, f_name='../ankur/New_Figures/Human_Knee_Cartilage/' + self.method + '.svg')
# 		return matches

# 	def get_matching_edges(self, plot=True, f_name='plot.png'):
# 		df_temp = pd.DataFrame(columns=['xt', 'matches'])

# 		g1 = corr2adjlist(corr=self.network1, top=self.top)
# 		g2 = corr2adjlist(corr=self.network2)

# 		for xt in range(0, g2.shape[0], self.dx):
# 			e1 = np.array(g1[:, :2], dtype=np.int32).transpose().tolist()
# 			e2 = np.array(g2[:, :2], dtype=np.int32)[xt:xt + self.dx, :].transpose().tolist()

# 			df_temp = df_temp.append({'xt': xt, 'matches': (len(set(zip(*e1)) & set(zip(*e2))))}, ignore_index=True)

# 		df_match = df_temp.copy()
# 		df_match['method'] = self.method
# 		self.df_global = self.df_global.append(df_match, ignore_index=True)
# 		if plot:
# 			self.plot_matching_edges(df_match, f_name)
# 		return df_temp


# 	def plot_matching_edges(self, df_match, f_name='plot.png'):
# 		df_match['total_matches'] = np.cumsum(df_match.matches)
# 		df_match['y'] = (df_match['total_matches'] / self.top) * 100

# 		# sns.set(style="whitegrid")
# 		plt.figure(figsize=(10, 6))

# 		method_name = self.method + " based : "
# 		p = sns.lineplot(data=df_match, x="xt", y="y", label=self.method)
# 		p.set(xlabel="edges in osteoarthritis data",
# 			  ylabel="matches in control data (in %)",
# 		      title=method_name + "Edge overlapping " + self.network1['_label'] + " and " + self.network2['_label'])
# 		plt.savefig(f_name)
# 		plt.show()

# 		# Calculate and print AUC
# 		auc = integrate.simps(x=df_match['xt'], y=df_match['total_matches'])
# 		max_auc = integrate.simps(x=list(df_match.xt), y=len(list(df_match.total_matches)) * [max(list(df_match.total_matches))])
# 		print(auc / max_auc)





from .MF_Evaluator import MF_Evaluator
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import integrate
from MF_Evaluation.helper import corr2adjlist
import warnings
warnings.filterwarnings('ignore')


class Eval_Knee_Cartilage(MF_Evaluator):
	def __init__(self, network1, network2, method=None, top=10000, dx=500) -> None:
		super().__init__(network1, network2, method)

		self.df_global = pd.DataFrame(columns=['xt', 'matches', 'method'])
		self.top = top
		self.dx = dx

	def _set(self, network1, network2=None):
		self.network1 = network1
		self.network2 = network2

	def evaluate(self):
		matches = self.get_matching_edges(plot=True, f_name='../ankur/New_Figures/Human_Knee_Cartilage/' + self.method + '.svg')
		return matches

	def get_matching_edges(self, plot=True, f_name='plot.png'):
		df_temp = pd.DataFrame(columns=['xt', 'matches'])

		g1 = corr2adjlist(corr=self.network1, top=self.top)
		g2 = corr2adjlist(corr=self.network2)

		for xt in range(0, g2.shape[0], self.dx):
			e1 = np.array(g1[:, :2], dtype=np.int32).transpose().tolist()
			e2 = np.array(g2[:, :2], dtype=np.int32)[xt:xt + self.dx, :].transpose().tolist()

			df_temp = df_temp.append({'xt': xt, 'matches': (len(set(zip(*e1)) & set(zip(*e2))))}, ignore_index=True)

		df_match = df_temp.copy()
		df_match['method'] = self.method
		self.df_global = self.df_global.append(df_match, ignore_index=True)
		if plot:
			self.plot_matching_edges(df_match, f_name)
		return df_temp


	def plot_matching_edges(self, df_match, f_name='plot.png'):
		df_match['total_matches'] = np.cumsum(df_match.matches)
		df_match['y'] = (df_match['total_matches'] / self.top) * 100

		# sns.set(style="whitegrid")
		plt.figure(figsize=(10, 6))

		method_name = self.method + " based : "
		p = sns.lineplot(data=df_match, x="xt", y="y", label=self.method)
		p.set(xlabel="edges in osteoarthritis data",
			  ylabel="matches in control data (in %)",
		      title=method_name + "Edge overlapping " + self.network1['_label'] + " and " + self.network2['_label'])
		# plt.savefig(f_name)
		plt.show()

		# Calculate and print AUC
		auc = integrate.simps(x=df_match['xt'], y=df_match['total_matches'])
		max_auc = integrate.simps(x=list(df_match.xt), y=len(list(df_match.total_matches)) * [max(list(df_match.total_matches))])
		print(auc / max_auc)