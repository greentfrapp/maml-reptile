import connexion
import six
import tensorflow as tf
import numpy as np

from swagger_server.models.predictions import Predictions  # noqa: E501
from swagger_server.models.predictions_datapoints import PredictionsDatapoints
from swagger_server.models.predictions_losses import PredictionsLosses
from swagger_server.models.predictions_predictions import PredictionsPredictions
from swagger_server.models.task import Task  # noqa: E501
from swagger_server import util

from swagger_server.controllers.Baseline import BaselineModel
from swagger_server.controllers.MAML import MAMLModel
from swagger_server.controllers.FOMAML import FOMAMLModel
from swagger_server.controllers.Reptile import ReptileModel
from swagger_server.controllers.SineTaskGenerator import SineTask


def step(task):  # noqa: E501
	"""Single step of gradient descent

	Single step of gradient descent # noqa: E501

	:param task: Sine Task Parameters
	:type task: dict | bytes

	:rtype: Predictions
	"""
	if connexion.request.is_json:
		task = Task.from_dict(connexion.request.get_json())  # noqa: E501
	task = SineTask(task.amplitude, task.phase)
	x, y = task.next(_k)
	eval_x, eval_y = task.eval()
	losses = {}
	predictions = {}
	for model_name, model in _models.items():
		model_loss, model_prediction = model.test(x=x, y=y, test_x=eval_x, test_y=eval_y, amplitude=task.amplitude)
		losses[model_name] = model_loss
		predictions[model_name] = model_prediction

	output = Predictions()
	
	output.datapoints = []
	for i, _ in enumerate(x):
		datapoint = PredictionsDatapoints()
		datapoint.x = float(x[i])
		datapoint.y = float(y[i])
		output.datapoints.append(datapoint)
	
	output.evaluation = []
	for i, _ in enumerate(eval_x):
		datapoint = PredictionsDatapoints()
		datapoint.x = float(eval_x[i])
		datapoint.y = float(eval_y[i])
		output.evaluation.append(datapoint)
	
	for model_name, loss in losses.items():
		datapoints = []
		for i, _ in enumerate(loss):
			datapoint = PredictionsDatapoints()
			datapoint.x = i
			datapoint.y = float(loss[i])
			datapoints.append(datapoint)
		losses[model_name] = datapoints
	prediction_losses = PredictionsLosses()
	prediction_losses.baseline = losses["baseline"]
	prediction_losses.maml = losses["maml"]
	prediction_losses.fomaml = losses["fomamlv2"]
	prediction_losses.reptile0 = losses["reptile0"]
	prediction_losses.reptile9 = losses["reptile9"]
	output.losses = prediction_losses

	for model_name, prediction in predictions.items():
		for i, step in enumerate(prediction):
			datapoints = []
			for j, _ in enumerate(eval_x):
				datapoint = PredictionsDatapoints()
				datapoint.x = float(eval_x[j])
				datapoint.y = float(step[j])
				datapoints.append(datapoint)
			predictions[model_name][i] = datapoints
	predictions_predictions = PredictionsPredictions()
	predictions_predictions.baseline = predictions["baseline"]
	predictions_predictions.maml = predictions["maml"]
	predictions_predictions.fomaml = predictions["fomamlv2"]
	predictions_predictions.reptile0 = predictions["reptile0"]
	predictions_predictions.reptile9 = predictions["reptile9"]
	output.predictions = predictions_predictions
	
	return output

_k = 10

_sess = tf.Session()
_models = {
	# "baseline": BaselineModel(),
	"maml": MAMLModel(name="maml", sess=_sess),
	"fomamlv1": MAMLModel(name="fomamlv1", sess=_sess, fo=True),
	"fomamlv2": FOMAMLModel(name="fomamlv2", sess=_sess),
	"reptile0": ReptileModel(name="reptile0", sess=_sess, beta1=0),
	"reptile9": ReptileModel(name="reptile9", sess=_sess, beta1=0),
}
_sess.run(tf.global_variables_initializer())
tf.train.Saver().restore(_sess, save_path="swagger_server/assets/model_10k/")
_models["baseline"] = BaselineModel()
