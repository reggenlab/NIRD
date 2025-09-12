from abc import ABC, abstractmethod


class MF_Evaluator(ABC):

	def __init__(self, network1, network2, method) -> None:
		super().__init__()

		self.network1 = network1
		self.network2 = network2

		self.method = method


	@abstractmethod
	def _set(self, network1, network2):
		pass

	@abstractmethod
	def evaluate(self):
		pass