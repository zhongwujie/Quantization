import torch
import torch.utils.data
import torchvision
import os
import numpy as np
from torch import nn
from torchvision.models import vgg16, VGG16_Weights
from torchvision import transforms

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self, name, fmt=':f'):
		self.name = name
		self.fmt = fmt
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

	def __str__(self):
		fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
		return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].reshape(-1).float().sum(0, keepdim = True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res


def evaluate(model, criterion, data_loader, neval_batches, verbose = True):
	model.eval()
	top1 = AverageMeter('Acc@1', ':6.2f')
	top5 = AverageMeter('Acc@5', ':6.2f')
	cnt = 0
	with torch.no_grad():
		for image, target in data_loader:
			output = model(image)
			loss = criterion(output, target)
			acc1, acc5 = accuracy(output, target, topk=(1, 5))
			if verbose:
				print("batch: {}, acc1: {:.2f}".format(cnt, acc1.item()))
			top1.update(acc1[0], image.size(0))
			top5.update(acc5[0], image.size(0))
			cnt += 1
			if cnt >= neval_batches:
				return top1, top5

	return top1, top5


def load_imagenet(data_path, batch_size):
	train_dir = os.path.join(data_path, "ILSVRC/Data/CLS-LOC/train")
	val_dir = os.path.join(data_path, "ILSVRC/Data/CLS-LOC/val_sorted")
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Resize([224, 224]),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
		shuffle=True, num_workers=0)

	testset = torchvision.datasets.ImageFolder(root=val_dir, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
		shuffle=False, num_workers=0)
	return trainloader, testloader


def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
	model.train()
	top1 = AverageMeter('Acc@1', ':6.2f')
	top5 = AverageMeter('Acc@5', ':6.2f')
	avgloss = AverageMeter('Loss', '1.5f')

	cnt = 0
	for image, target in data_loader:
		cnt += 1
		image, target = image.to(device), target.to(device)
		output = model(image)
		loss = criterion(output, target)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		acc1, acc5 = accuracy(output, target, topk=(1, 5))
		top1.update(acc1[0], image.size(0))
		top5.update(acc5[0], image.size(0))
		avgloss.update(loss, image.size(0))
		if cnt >= ntrain_batches:
			print('Loss: {:.2f}'.format(avgloss.avg))

			print('Accuracy: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
					.format(top1=top1, top5=top5))
			return

	print('Full imagenet train set:  * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
			.format(top1=top1, top5=top5))
	return


def load_model(model_file, model):
	state_dict = torch.load(model_file)
	model.load_state_dict(state_dict)
	model.to('cpu')
	return model


'''User defined conv module'''
class usr_conv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, weights, bias) -> None:
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.weights = weights
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
		self.conv.weight.data.fill_(weights)
		self.conv.bias.data.fill_(bias)
	
	def forward(self, x):
		output = self.conv(x)
		return output


'''matrix1 slides on matrix2'''
def perform_conv(matrix1: np.ndarray, matrix2: np.ndarray):
	return np.convolve(matrix2, matrix1, "same")

def _set_module(model, submodule_key, module):
	tokens = submodule_key.split('.')
	number = int(tokens[-1])
	print(model)
	# print(model.get_submodule("0"))
	# layer = getattr(model._modules, number)
	# print(layer)
	


'''Replace the conv and fc layer, and then evaluate the model'''
def replace_evaluate(data_path, batch_size, neval_batches, criterion, model):
	print("====== begin replace ======")
	conv_names = [k for k, m in model.named_modules() if m.original_name == 'Conv2d']
	for conv_name in conv_names:
		new_layer = usr_conv(1, 5, 0, 0, 0)
		_set_module(model, conv_name, new_layer)
