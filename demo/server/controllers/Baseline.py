import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import Callback
from keras.losses import mean_squared_error
from keras.optimizers import Adam
from keras import backend as K


class LossHistory(Callback):

	def on_train_begin(self, logs={}):
		self.losses = []

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))

class BaselineModel(object):

	def __init__(self):
		self.hidden_1 = 64
		self.hidden_2 = 64
		self.model = self.build_model()

	def build_model(self):
		model = Sequential()
		model.add(Dense(self.hidden_1, input_shape=(1,), activation='relu'))
		model.add(Dense(self.hidden_2, activation='relu'))
		model.add(Dense(1))
		self.weights = model.get_weights()
		model.compile(optimizer=Adam(lr=1e-2), loss='mean_squared_error')
		return model

	def fit(self, x, y, amplitude, eval_x=None, eval_y=None, grad_steps=1, return_losses=False):
		losses = []
		predictions = []
		if return_losses:
			losses.append(self.calculate_loss(eval_x, eval_y, amplitude))
			predictions.append(self.model.predict(eval_x))
		for i in np.arange(grad_steps):
			self.model.fit(x, y,
				batch_size=len(x),
				epochs=1,
				validation_split=0.,
				verbose=0,
			)
			if return_losses:
				losses.append(self.calculate_loss(eval_x, eval_y, amplitude))
				predictions.append(self.model.predict(eval_x))
		if return_losses:
			return losses, predictions

	def predict(self, x):
		return self.model.predict(x)

	def calculate_loss(self, x, y, amplitude):
		return np.mean((y / amplitude - self.predict(x) / amplitude) ** 2)

	def test(self, x, y, test_x, test_y, amplitude, grad_steps=32):
		self.model.set_weights(self.weights)
		losses, predictions = self.fit(x=x, y=y, eval_x=test_x, eval_y=test_y, grad_steps=grad_steps, amplitude=amplitude, return_losses=True)
		return losses, predictions




