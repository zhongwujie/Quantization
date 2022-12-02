'''
@reference: https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/tutorials/quant_resnet50.html
@dataset: imagenet
'''
import torch
import torch.utils.data
import torchvision
import sys
import os
from torch import nn
from tqdm import tqdm

from torchvision.models import vgg16, VGG16_Weights
from torchvision import transforms
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

from utils import *

# from pytorch_quantization import quant_modules
# quant_modules.initialize()

# Calibration
# quant_desc_input = QuantDescriptor(calib_method='histogram')
# quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
# quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
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

def collect_stats(model, data_loader, num_batches):
  """Feed data to the network and collect statistic"""

  # Enable calibrators
  for name, module in model.named_modules():
    if isinstance(module, quant_nn.TensorQuantizer):
      if module._calibrator is not None:
        module.disable_quant()
        module.enable_calib()
      else:
        module.disable()

  for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
    model(image)
    if i >= num_batches:
      break

  # Disable calibrators
  for name, module in model.named_modules():
    if isinstance(module, quant_nn.TensorQuantizer):
      if module._calibrator is not None:
        module.enable_quant()
        module.disable_calib()
      else:
        module.enable()

def compute_amax(model, **kwargs):
  # Load calib result
  for name, module in model.named_modules():
    if isinstance(module, quant_nn.TensorQuantizer):
      if module._calibrator is not None:
        if isinstance(module._calibrator, calib.MaxCalibrator):
          module.load_calib_amax()
        else:
          module.load_calib_amax(**kwargs)
      print(F"{name:40}: {module}")

# # It is a bit slow since we collect histograms on CPU
# with torch.no_grad():
#   collect_stats(model, trainloader, num_batches=2)
#   compute_amax(model, method="percentile", percentile=99.99)

'''
The number of image is determined by batch_size and number of batches
'''
def run_unquantized():
  model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
  batch_size = 250
  data_path = "~/code/dataset/imagenet"
  trainloader, testloader = load_imagenet(data_path, batch_size)
  criterion = nn.CrossEntropyLoss()
  num_eval_batches = 20
  print("num_eval_batches:{}, batch_size:{}".format(num_eval_batches, batch_size))
  with torch.no_grad():
    top1, top5 = evaluate(model, criterion, testloader, neval_batches = num_eval_batches)
    print('number of image %d :Evaluation accuracy: %2.2f'%(batch_size * num_eval_batches, top1.avg))


if __name__ == "__main__":
  run_unquantized()