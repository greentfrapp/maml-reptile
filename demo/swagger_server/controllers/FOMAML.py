import tensorflow as tf
import numpy as np

from swagger_server.controllers.utils import update_target_graph


class FOMAMLModel(object):

	def __init__(self, name, sess, grad_steps=32, beta1=0):
		self.name = name
		with tf.variable_scope(self.name):
			self.hidden_1 = 64
			self.hidden_2 = 64
			self.meta_trainer = tf.train.AdamOptimizer(learning_rate=1e-3)
			self.beta1 = beta1
			self.grad_steps = grad_steps
			self.sess = sess
			self.build_model()
		

	def build_model(self):
		self.inputs = tf.placeholder(
			shape=[None, 1],
			dtype=tf.float32,
		)
		self.labels = tf.placeholder(
			shape=[None, 1],
			dtype=tf.float32,
		)
		self.task_amplitude = tf.placeholder(
			shape=None,
			dtype=tf.float32,
		)
		self.ep = tf.Variable(
			0, 
			dtype=tf.int32, 
			name='episodes',
			trainable=False
		)
		self.inc_ep = self.ep.assign_add(1)

		network_names = ["meta", "learner"]
		self.outputs = {}
		for name in network_names:
			with tf.variable_scope(name):
				dense_1 = tf.layers.dense(
					inputs=self.inputs,
					units=self.hidden_1,
					activation=tf.nn.relu,
					kernel_initializer=tf.truncated_normal_initializer(.0,.01),
					name="dense_1",
				)
				dense_2 = tf.layers.dense(
					inputs=dense_1,
					units=self.hidden_2,
					activation=tf.nn.relu,
					kernel_initializer=tf.truncated_normal_initializer(.0,.01),
					name="dense_2",
				)
				self.outputs[name] = tf.layers.dense(
					inputs=dense_2,
					units=1,
					activation=None,
					kernel_initializer=tf.truncated_normal_initializer(.0,.01),
					name="output",
				)

		self.loss = tf.losses.mean_squared_error(self.labels / self.task_amplitude, self.outputs["learner"] / self.task_amplitude)

		self.optimize = tf.train.AdamOptimizer(learning_rate=1e-2, beta1=self.beta1).minimize(self.loss)
		self.fresh_optimize = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(self.loss)

		local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
		self.learner_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "{}/learner".format(self.name))
		self.meta_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "{}/meta".format(self.name))
		self.fomaml_grad = tf.gradients(self.loss, self.learner_vars)
		# self.fomaml_grad, _ = tf.clip_by_global_norm(self.fomaml_grad, 40.0)
		self.update_meta = self.meta_trainer.apply_gradients(zip(self.fomaml_grad, self.meta_vars))
		self.copy_meta_to_learner = update_target_graph("{}/meta".format(self.name), "{}/learner".format(self.name))

	def fit(self, x, y, test_x=[], test_y=[], test=False, return_losses=False, amplitude=1.):
		losses = []
		outputs = []
		if return_losses:
			losses.append(self.sess.run(self.loss, feed_dict={self.inputs: test_x, self.labels: test_y, self.task_amplitude: amplitude}))
			outputs.append(self.predict(test_x))
		if test:
			for i in np.arange(self.grad_steps):
				self.sess.run(self.fresh_optimize, feed_dict={self.inputs: x, self.labels: y, self.task_amplitude: amplitude})
				if return_losses:
					losses.append(self.sess.run(self.loss, feed_dict={self.inputs: test_x, self.labels: test_y, self.task_amplitude: amplitude}))
					outputs.append(self.predict(test_x))
		else:
			for i in np.arange(50):
				self.sess.run(self.optimize, feed_dict={self.inputs: x, self.labels: y, self.task_amplitude: amplitude})
				if return_losses:
					losses.append(self.sess.run(self.loss, feed_dict={self.inputs: test_x, self.labels: test_y, self.task_amplitude: amplitude}))
					outputs.append(self.predict(test_x))
		if return_losses:
			return losses, outputs

	def predict(self, x):
		return self.sess.run(self.outputs["learner"], feed_dict={self.inputs: x})

	def train(self, x, y, amplitude):
		self.sess.run(self.copy_meta_to_learner)
		print("Meta-training FOMAML {} Task #{}...".format(self.name, self.sess.run(self.ep)))
		self.fit(x=x, y=y, amplitude=amplitude)
		print(self.sess.run(self.loss, feed_dict={self.inputs: x, self.labels: y, self.task_amplitude: amplitude}))
		self.sess.run(self.update_meta, feed_dict={self.inputs: x, self.labels: y, self.task_amplitude: amplitude})
		self.sess.run(self.inc_ep)

	def test(self, x, y, test_x, test_y, amplitude):
		self.sess.run(self.copy_meta_to_learner)
		losses, predictions = self.fit(x=x, y=y, test_x=test_x, test_y=test_y, amplitude=amplitude, test=True, return_losses=True)
		# prediction = self.predict(test_x)
		return losses, predictions
		
