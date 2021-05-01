import tensorflow as tf


class SineRegressionNetwork(object):
	
	def __init__(self, name):
		super(SineRegressionNetwork, self).__init__()
		self.name = name
		with tf.variable_scope(self.name):
			self.build_model()
			self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self._placeholders = [tf.placeholder(v.dtype.base_dtype, shape=v.get_shape()) for v in self.weights]
			assigns = [tf.assign(v, p) for v, p in zip(self.weights, self._placeholders)]
			self._assign_op = tf.group(*assigns)

	def build_model(self):
		# need to define
		#	self.input
		#	self.labels
		#	self.output
		#	self.loss
		self.input = tf.placeholder(
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
		self.output = tf.layers.dense(
			inputs=dense_2,
			units=1,
			activation=None,
			kernel_initializer=tf.truncated_normal_initializer(.0,.01),
			name="output",
		)

		self.loss = tf.losses.mean_squared_error(self.labels / self.task_amplitude, self.output / self.task_amplitude)
		
	def import_weights(self, weights):
		self._session.run(self._assign_op, feed_dict=dict(zip(self._placeholders, weights)))

	def copy(self, name, weights):
		new_network = SineRegressionNetwork(name)
		new_network.import_weights(weights)
		return new_network
