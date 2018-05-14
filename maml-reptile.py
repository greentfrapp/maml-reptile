import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
import json

from SineTaskGenerator import SineTaskGenerator
# from Baseline import BaselineModel
from MAML import MAMLModel
from FOMAML import FOMAMLModel
from Reptile import ReptileModel


if __name__ == "__main__":
	task_dist = SineTaskGenerator()
	k = 10
	grad_steps = 32
	episodes = 10000

	with tf.Session() as sess:
		models = {
			"maml": MAMLModel(name="maml", sess=sess),
			"fomamlv1": MAMLModel(name="fomamlv1", sess=sess, fo=True),
			"fomamlv2": FOMAMLModel(name="fomamlv2", sess=sess),
			"reptile": ReptileModel(name="reptile", sess=sess),
		}
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		
		losses = {
			"maml": [],
			"fomamlv1": [],
			"fomamlv2": [],
			"reptile": [],
		}
		for i in np.arange(episodes):
			task = task_dist.new_task()
			x, y = task.next(k * 3)
			for model_name, model in models.items():
				losses[model_name].append(float(model.train(x=x, y=y, amplitude=task.amplitude)))
		saver.save(sess, save_path="./temp/")
		with open("./temp/losses.json", 'w') as file:
			json.dump(losses, file)

		n_tests = 100
		mean_losses = {}
		outputs = {}
		for i in np.arange(n_tests):
			print("Testing with Task #{}".format(i + 1))
			test = task_dist.new_task()
			x, y = test.next(k)
			eval_x, eval_y = test.eval()
			for model_name, model in models.items():
				if model_name not in mean_losses:
					mean_losses[model_name] = []
				model_loss, model_output = model.test(x=x, y=y, test_x=eval_x, test_y=eval_y, amplitude=test.amplitude)
				mean_losses[model_name].append(model_loss)
				outputs[model_name] = model_output[-1]
		for model_name, mean_loss in mean_losses.items():
			mean_losses[model_name] = np.mean(mean_loss, axis=0)

		fig, ax = plt.subplots()
		for model_name, mean_loss in mean_losses.items():
			ax.plot(np.arange(grad_steps), mean_loss, label=model_name)
		ax.legend()

		fig, ax2 = plt.subplots()
		ax2.scatter(x, y)
		ax2.plot(eval_x, eval_y, label="Truth")
		for model_name, model_output in outputs.items():
			ax2.plot(eval_x, model_output, label=model_name)
		ax2.legend()

		plt.show()
