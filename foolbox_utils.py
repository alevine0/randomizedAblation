import numpy as np
import foolbox
import scipy.stats
import torch
import utils
class MisclassificationOrAbstain(foolbox.criteria.Criterion):

	def __init__(self, num_samples, alpha):
		super(MisclassificationOrAbstain, self).__init__()
		self.alpha = alpha
		self.num_samples = num_samples

	def name(self):
		return '{}-alpha-{:.04f}-samples-{}'.format(self.__class__.__name__, self.alpha, str(self.num_samples))

	def is_adversarial(self, predictions, label):
		sorte = np.argsort(predictions)
		topidx = sorte[-1]
		if (topidx != label):
			return True
		secondidx = sorte[-2]
		topcount = predictions[topidx]*self.num_samples
		secondcount = predictions[secondidx]*self.num_samples
		return scipy.stats.binom_test(topcount,topcount+secondcount, .5) > self.alpha

class AblatedTorchModel(foolbox.models.Model):
	def __init__(
			self,
			model,
			num_samples,
			keep,
			bounds,
			num_classes,
			channel_axis=1,
			device=None,
			preprocessing=(0, 1)):
		super(AblatedTorchModel, self).__init__(bounds=bounds,
										   channel_axis=channel_axis,
										   preprocessing=preprocessing)

		self.num_classes = num_classes

		if device is None:
			self.device = torch.device(
				"cuda:0" if torch.cuda.is_available() else "cpu")
		elif isinstance(device, str):
			self.device = torch.device(device)
		else:
			self.device = device
		self.model = model.to(self.device)
		self.num_samples=num_samples
		self.keep = keep
	def batch_predictions(self, inputs):
		with torch.no_grad():
			inputs, _ = self._process_input(inputs)
			inputs = torch.from_numpy(inputs).to(self.device)
			predictions = utils.avg_hard_forward(inputs, self.model, self.num_samples, self.keep)
			predictions = predictions.detach().cpu().numpy()
		return predictions
	def num_classes(self):
		return self.num_classes
