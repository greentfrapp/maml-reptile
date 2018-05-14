import tensorflow as tf
import numpy as np

from utils import update_target_graph


class MAMLModel(object):

	def __init__(self, name, sess, grad_steps=32, fo=False):
		self.name = name
		with tf.variable_scope(self.name):
			self.hidden_1 = 64
			self.hidden_2 = 64
			self.inner_lr = 1e-2
			self.meta_lr = 1e-3
			self.is_fo = fo
			self.grad_steps = grad_steps
			self.weights = self.build_model()
			self.build_ops()
			self.sess = sess
		

	def build_model(self):

		self.inputs = tf.placeholder(
			shape=[None, 1],
			dtype=tf.float32,
		)
		self.eval_inputs = tf.placeholder(
			shape=[None, 1],
			dtype=tf.float32,
		)
		self.labels = tf.placeholder(
			shape=[None, 1],
			dtype=tf.float32,
		)
		self.eval_labels = tf.placeholder(
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

		weights = {}
		
		# hidden_1
		weights['w_hidden_1'] = tf.Variable(tf.truncated_normal([1, self.hidden_1], stddev=0.01), dtype=tf.float32)
		weights['b_hidden_1'] = tf.Variable(tf.zeros([self.hidden_1]), dtype=tf.float32)
		
		# hidden_2
		weights['w_hidden_2'] = tf.Variable(tf.truncated_normal([self.hidden_1, self.hidden_2], stddev=0.01, dtype=tf.float32))
		weights['b_hidden_2'] = tf.Variable(tf.zeros([self.hidden_2]), dtype=tf.float32)
		
		# output
		weights['w_output'] = tf.Variable(tf.truncated_normal([self.hidden_2, 1], stddev=0.01), dtype=tf.float32)
		weights['b_output'] = tf.Variable(tf.zeros([1]), dtype=tf.float32)

		return weights

	def forwardprop(self, x, weights):
		hidden = x
		for i in range(2):
			hidden = tf.nn.relu(tf.matmul(hidden, weights['w_hidden_{}'.format(i + 1)]) + weights['b_hidden_{}'.format(i + 1)])
		output = tf.matmul(hidden, weights['w_output']) + weights['b_output']
		return output

	def build_ops(self):
		self.eval_losses = []
		self.eval_outputs = []
		key_order = ["w_hidden_1", "b_hidden_1", "w_hidden_2", "b_hidden_2", "w_output", "b_output"]
		loss = tf.losses.mean_squared_error(self.labels / self.task_amplitude, self.forwardprop(self.inputs, self.weights) / self.task_amplitude)
		eval_loss = tf.losses.mean_squared_error(self.eval_labels / self.task_amplitude, self.forwardprop(self.eval_inputs, self.weights) / self.task_amplitude)
		self.eval_losses.append(eval_loss)
		grads = tf.gradients(loss, list(self.weights.values()))
		if self.is_fo:
			grads = [tf.stop_gradient(grad) for grad in grads]
		grads, _ = tf.clip_by_global_norm(grads,40.0)
		grads = dict(zip(self.weights.keys(), grads))
		fast_weights = dict(zip(self.weights.keys(), [weight - self.inner_lr * grads[key] for key, weight in self.weights.items()]))
		self.eval_outputs.append(self.forwardprop(self.eval_inputs, fast_weights))
		for i in np.arange(self.grad_steps - 1):
			loss = tf.losses.mean_squared_error(self.labels, self.forwardprop(self.inputs, fast_weights))
			eval_loss = tf.losses.mean_squared_error(self.eval_labels / self.task_amplitude, self.forwardprop(self.eval_inputs, fast_weights) / self.task_amplitude)
			self.eval_losses.append(eval_loss)
			grads = tf.gradients(loss, list(fast_weights.values()))
			if self.is_fo:
				grads = [tf.stop_gradient(grad) for grad in grads]
			grads, _ = tf.clip_by_global_norm(grads,40.0)
			grads = dict(zip(fast_weights.keys(), grads))
			fast_weights = dict(zip(fast_weights.keys(), [weight - self.inner_lr * grads[key] for key, weight in fast_weights.items()]))
			self.eval_outputs.append(self.forwardprop(self.eval_inputs, fast_weights))
		output = self.forwardprop(self.inputs, fast_weights)
		loss = tf.losses.mean_squared_error(self.labels / self.task_amplitude, output / self.task_amplitude)
		self.loss = loss
		self.output = output
		optimizer = tf.train.AdamOptimizer(self.meta_lr)
		self.metatrain_op = optimizer.minimize(loss)

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
