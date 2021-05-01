import tensorflow as tf
import numpy as np

from utils import update_target_graph


class MAMLModel(object):

	def __init__(self, name, sess, network, grad_steps=32, fo=False):
		self.name = name
		self.sess = sess
		self.network = network
		self.grad_steps = grad_steps
		self.is_fo = fo
		self.build_ops()

	def build_ops(self):
		gradient = tf.gradients(self.network.loss, list(self.network.weights.values()))
		if self.is_fo:
			gradient = [tf.stop_gradient(grad) for grad in gradient]
		# gradient, _ = tf.clip_by_global_norm(gradient, 40.0)
		gradient = dict(zip(self.network.weights.keys(), gradient))
		fast_weights = dict(zip(self.network.weights.keys(), [weight - self.inner_lr * gradient[key] for key, weight in self.network.weights.items()]))
		network = self.network.copy(fast_weights)
		for i in np.arange(self.grad_steps - 1):
			gradient = tf.gradients(network.loss, list(fast_weights.values()))
			if self.is_fo:
				gradient = [tf.stop_gradient(grad) for grad in gradient]
			# gradient, _ = tf.clip_by_global_norm(gradient, 40.0)
			gradient = dict(zip(fast_weights.keys(), gradient))
			fast_weights = dict(zip(fast_weights.keys(), [weight - self.inner_lr * gradient[key] for key, weight in fast_weights.items()]))
			network = self.network.copy(fast_weights)
		self.maml_optimize = tf.train.AdamOptimizer(self.meta_lr).minimize(network.loss)

	def predict(self, x):
		return self.sess.run(self.forwardprop(self.inputs, self.weights), feed_dict={self.inputs: x})

	def train(self, x, y, amplitude):
		loss, _ = self.sess.run([self.loss, self.metatrain_op], feed_dict={self.inputs: x, self.labels: y, self.task_amplitude: amplitude})
		if self.sess.run(self.ep) % 50 == 0:
			print("Meta-training MAML {} Task #{}...".format(self.name, self.sess.run(self.ep)))
			print(loss)
		self.sess.run(self.inc_ep)
		return loss

	def test(self, x, y, test_x, test_y, amplitude):
		losses, predictions = self.sess.run([self.eval_losses, self.eval_outputs], feed_dict={self.inputs: x, self.labels: y, self.eval_inputs: test_x, self.eval_labels: test_y, self.task_amplitude: amplitude})
		return losses, predictions
