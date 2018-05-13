import numpy as np


class SineTaskGenerator(object):

	def new_task(self):
		amplitude = np.random.uniform(0.1, 5.)
		phase = np.random.uniform(0, 2 * np.pi)
		return SineTask(amplitude, phase)

class SineTask(object):

	def __init__(self, amplitude, phase):
		self.amplitude = amplitude
		self.phase = phase

	def next(self, k):
		x = np.random.rand(k) * 10 - 5 
		y = self.amplitude * np.sin(x + self.phase)
		return x.reshape(-1, 1), y.reshape(-1, 1)

	def get_y(self, x):
		return self.amplitude * np.sin(x + self.phase)

	def eval(self):
		eval_x = np.arange(-5., 5.1, 10./49.).reshape(-1, 1)
		return eval_x, self.get_y(eval_x)
