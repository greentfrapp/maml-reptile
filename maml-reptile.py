import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
import json
from tensorflow.python.platform import flags

from SineTaskGenerator import SineTaskGenerator
# from Baseline import BaselineModel
from MAML import MAMLModel
from FOMAML import FOMAMLModel
from Reptile import ReptileModel


FLAGS = flags.FLAGS

# General
flags.DEFINE_bool("train", False, "Whether to train")
flags.DEFINE_bool("test", False, "Whether to test")
flags.DEFINE_string("savepath", "temp", "Savepath")

# Train
flags.DEFINE_integer("train_gradsteps", 64, "Number of gradient steps during training")
flags.DEFINE_integer("train_samples", 30, "Number of training samples during training")
flags.DEFINE_integer("train_epochs", 100, "Number of training epochs")
flags.DEFINE_integer("train_totalepisodes", 1000, "Number of tasks generated for training")
flags.DEFINE_integer("train_batchsize", 10, "Training batch size")

# Test
flags.DEFINE_integer("test_gradsteps", None, "Number of gradient steps during testing (if different from training)")
flags.DEFINE_integer("test_samples", None, "Number of training samples during testing (if different from training)")
flags.DEFINE_bool("show_loss", False, "Whether to show graph of loss against gradient step")
flags.DEFINE_bool("show_pre", False, "Whether to show predicted function before training")
flags.DEFINE_bool("show_post", False, "Whether to show predicted function after training")


def main():
	tf.gfile.MakeDirs(FLAGS.savepath)

	task_dist = SineTaskGenerator()
	tasks = []
	for i in np.arange(FLAGS.train_totalepisodes):
		tasks.append(task_dist.new_task())

	with tf.Session() as sess:
		models = {
			# "maml": MAMLModel(name="maml", sess=sess),
			# "fomamlv1": MAMLModel(name="fomamlv1", sess=sess, fo=True),
			# "fomamlv2": FOMAMLModel(name="fomamlv2", sess=sess),
			"reptile": ReptileModel(
				name="reptile", 
				sess=sess,
				train_gradsteps=FLAGS.train_gradsteps,
				test_gradsteps=FLAGS.test_gradsteps or FLAGS.train_gradsteps,
				train_samples=FLAGS.train_samples,
				test_samples=FLAGS.test_samples or FLAGS.train_samples,
			),
		}
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		if FLAGS.test:
			saver.restore(sess, save_path=os.path.join(FLAGS.savepath, "./"))
		
		if FLAGS.train:
			losses = {
				# "maml": [],
				# "fomamlv1": [],
				# "fomamlv2": [],
				"reptile": [],
			}
			n_batches = FLAGS.train_totalepisodes / FLAGS.train_batchsize
			for i in np.arange(FLAGS.train_epochs):
				print("Epoch #{}...".format(i + 1))
				np.random.shuffle(tasks)
				for j in np.arange(int(n_batches)):
					print("\tBatch #{}...".format(j + 1))
					start = int(j * FLAGS.train_batchsize)
					end = int(start + FLAGS.train_batchsize)
					minibatch = tasks[start:end]
					for model_name, model in models.items():
						model.train(tasks=minibatch)

			# for i in np.arange(episodes):
			# 	task = task_dist.new_task()
			# 	x, y = task.next(k * 3)
			# 	for model_name, model in models.items():
			# 		losses[model_name].append(float(model.train(x=x, y=y, amplitude=task.amplitude)))
			# saver.save(sess, save_path=os.path.join(FLAGS.savepath, "./"))
			# with open(os.path.join(FLAGS.savepath, "loss.json"), 'w') as file:
			# 	json.dump(losses, file)

		n_tests = 1
		mean_losses = {}
		preoutputs = {}
		postoutputs = {}
		for i in np.arange(n_tests):
			print("Testing with Task #{}".format(i + 1))
			test = task_dist.new_task()
			x, y = test.next(FLAGS.test_samples or FLAGS.train_samples)
			eval_x, eval_y = test.eval()
			for model_name, model in models.items():
				if model_name not in mean_losses:
					mean_losses[model_name] = []
				model_loss, model_output = model.test(x=x, y=y, test_x=eval_x, test_y=eval_y, amplitude=test.amplitude)
				mean_losses[model_name].append(model_loss)
				preoutputs[model_name] = model_output[0]
				postoutputs[model_name] = model_output[-1]
		for model_name, mean_loss in mean_losses.items():
			mean_losses[model_name] = np.mean(mean_loss, axis=0)

		# Loss Plot
		if FLAGS.show_loss:
			_, ax_loss = plt.subplots()
			for model_name, mean_loss in mean_losses.items():
				ax_loss.plot(np.arange(grad_steps), mean_loss, label=model_name)
			ax_loss.legend()
			ax_loss.set_title("Losses")

		# Pretraining Function
		if FLAGS.show_pre:
			_, ax_pre = plt.subplots()
			ax_pre.scatter(x, y)
			ax_pre.plot(eval_x, eval_y, label="Truth")
			for model_name, model_output in preoutputs.items():
				ax_pre.plot(eval_x, model_output, label=model_name)
			ax_pre.legend()
			ax_pre.set_title("Before Training")

		# Posttraining Function
		if FLAGS.show_post:
			_, ax_post = plt.subplots()
			ax_post.scatter(x, y)
			ax_post.plot(eval_x, eval_y, label="Truth")
			for model_name, model_output in postoutputs.items():
				ax_post.plot(eval_x, model_output, label=model_name)
			ax_post.legend()
			ax_post.set_title("After Training")

		if FLAGS.show_loss or FLAGS.show_pre or FLAGS.show_post:
			plt.show()


if __name__ == "__main__":
	main()
