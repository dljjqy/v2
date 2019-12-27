import torch.nn as nn
import torch


class MultiClassBatchDiceLoss(nn.Module):
	'''
	dice loss computed in whole batch and classes, which consider area difference in each image
	'''
	def __init__(self):
		super(MultiClassBatchDiceLoss, self).__init__()
		self.PREPROCESS = nn.Softmax(dim = 1)

	def forward(self, output, target, weights = None):
		output = self.PREPROCESS(output)

		smooth = 1e-10
		target = torch.unsqueeze(target, 1)
		one_hot_target = torch.zeros_like(output).scatter_(1, target, 1)

		N = one_hot_target.shape[0]
		C = one_hot_target.shape[1]

		if weights is None:
			weights = torch.ones(1, C) * (1 / (C - 1))
			weights[0, 0] = 0  # background set to zero

		weights = torch.unsqueeze(weights, 2).type_as(output)

		output_flat = output.view(N, C, -1) * weights
		one_hot_target_flat = one_hot_target.view(N, C, -1) * weights

		intersection = output_flat * one_hot_target_flat

		loss = 1 - 2 * (intersection.sum()) / (output_flat.sum() + one_hot_target_flat.sum() + smooth)
		loss /= (N - 1)

		return loss

class DiceLoss(nn.Module):
	def __init__(self):
		super(DiceLoss, self).__init__()
		self.PREPROCESS = nn.Softmax(dim = 1)

	def forward(self, output, target):
		output = self.PREPROCESS(output)
		
		N = target.size(0)
		smooth = 1e-10

		output_flat = output.view(N, -1)
		target_flat = target.view(N, -1)

		intersection = output_flat * target_flat
		loss = 2 * (intersection.sum(1)) / (output_flat.sum(1) + target_flat.sum(1) + smooth)
		loss = 1 - loss.sum() / N

		return loss

class BatchCrossEntropyLoss(nn.Module):
	def __init__(self):
		super(BatchCrossEntropyLoss, self).__init__()
		self.Loss = nn.CrossEntropyLoss()

	def forward(self, images, masks):
		loss = 0
		for image, mask in zip(images, masks):
			loss += self.Loss(image, mask)

		return loss / len(images)