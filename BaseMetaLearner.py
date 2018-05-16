

class BaseMetaLearner(object):

	def __init__(self, model):
		super(BaseMetaLearner, self).__init__()

	def train(self, tasks):
		return None

	def test(self, tasks):
		return None
