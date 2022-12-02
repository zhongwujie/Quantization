'''
@reference: https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/tutorials/quant_resnet50.html
@dataset: imagenet
'''
import torch
import torch.utils.data
import time
from torch import nn
from tqdm import tqdm
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models.quantization import resnet50, ResNet50_QuantizedWeights
from torchvision.models.quantization import mobilenet_v2, MobileNet_V2_QuantizedWeights
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

from utils import *
from pytorch_quantization import quant_modules
# quant_modules.initialize()

"""Feed data to the network and collect statistic"""
def collect_stats(model, data_loader, num_batches):
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
  print("compute_amax")
  # Load calib result
  for name, module in model.named_modules():
    if isinstance(module, quant_nn.TensorQuantizer):
      if module._calibrator is not None:
        if isinstance(module._calibrator, calib.MaxCalibrator):
          module.load_calib_amax()
        else:
          module.load_calib_amax(**kwargs)
      print(F"{name:40}: {module}")


'''
@brief: post-train vgg16 and save it.
'''
def post_train_quantize(data_path, batch_size, num_eval_batches, criterion):
  quant_modules.initialize()
  model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
  model.eval()
  quant_desc_input = QuantDescriptor(calib_method='histogram')
  quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
  quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
  trainloader, testloader = load_imagenet(data_path, batch_size)
  with torch.no_grad():
    collect_stats(model, trainloader, num_batches = 4)
    compute_amax(model, method = "percentile", percentile = 99.99)
    print("num_eval_batches:{}, batch_size:{}".format(num_eval_batches, batch_size))
    top1, top5 = evaluate(model, criterion, testloader, neval_batches = num_eval_batches)
    print('number of image %d :Evaluation accuracy: %2.2f'%(batch_size * num_eval_batches, 
      top1.avg))
  torch.save(model.state_dict(), "./output/quant_vgg16.pth")


'''
@brief: evaluate the model, using imagenet
'''
def evaluate_model(data_path, batch_size, num_eval_batches, criterion, model = 
  vgg16(weights=VGG16_Weights.IMAGENET1K_V1)):
  model.eval()
  trainloader, testloader = load_imagenet(data_path, batch_size)
  print("num_eval_batches:{}, batch_size:{}".format(num_eval_batches, batch_size))
  with torch.no_grad():
    top1, top5 = evaluate(model, criterion, testloader, neval_batches = num_eval_batches)
    print('number of image %d :Evaluation accuracy: %2.2f'%(batch_size * num_eval_batches, 
      top1.avg))

''' evaluate models in the output'''
def evaluate_saved_model(data_path, batch_size, num_eval_batches, criterion):
  quant_modules.initialize()
  model = vgg16()
  state_dict = torch.load("./output/quant_vgg16.pth", map_location = torch.device('cpu'))
  model.load_state_dict(state_dict)
  save_quantized_params(model)
  # evaluate_model(data_path, batch_size, num_eval_batches, criterion, model)

def save_quantized_models():
  model1 = resnet50(weights=ResNet50_QuantizedWeights.IMAGENET1K_FBGEMM_V2, quantize=True)
  model2 = mobilenet_v2(weights=MobileNet_V2_QuantizedWeights.IMAGENET1K_QNNPACK_V1, 
    quantize=True)
  torch.save(model1.state_dict(), "./output/quant_resnet50.pth")
  torch.save(model2.state_dict(), "./output/quant_mobilenet_v2.pth")


def save_quantized_params(quantized_model):
  with open("./params_int.txt",'w') as f:
    for layer,param in quantized_model.state_dict().items(): # param is weight or bias(Tensor)         
      print(layer, type(param))
      f.write(layer)
      f.write('\n')
      f.write(str(param))
      break


def main():
  data_path = "~/code/dataset/imagenet"
  batch_size = 250
  num_eval_batches = 20
  criterion = nn.CrossEntropyLoss()
  evaluate_saved_model(data_path, batch_size, num_eval_batches, criterion)


if __name__ == "__main__":
  main()