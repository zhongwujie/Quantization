'''
@brief: dump all the parameters for conv and fc layers.
'''
import torch
import torch.nn as nn
import numpy as np
from quantization import *


'''
@input: 
  - tensor data, 4 dimensions: (N, C, H, W)
  - kernel size, 2 dimensions: (H, W)
@output:
  - 2 dimensions: (N * OH, OW)
'''
def im2col(data: torch.Tensor, kernel_size: tuple):
  data = data.to(torch.float32)
  unfold = nn.Unfold(kernel_size=kernel_size)
  result = unfold(data).permute(0, 2, 1)
  width = result.shape[2]
  result = result.reshape([-1, width]).to(torch.int8)
  return result


def im2col_test():
  input0 = torch.tensor([[[[1, 2, 0], [1, 1, 3], [0, 2, 2]], [[0, 2, 1], [0, 3, 2],
    [1, 1, 0]], [[1, 2, 1], [0, 1, 3], [3, 3, 2]]]])
  input = torch.concat((input0, input0))
  print("input shape: ", input.size())
  kernel = torch.tensor([[[[1, 1], [2, 2]], [[1, 1], [1, 1]], [[0, 1], [1, 0]]],
    [[[1, 0], [0, 1]], [[2, 1], [2, 1]], [[1, 2], [2, 0]]]])
  kernel_shape = (2, 2)
  input_matrix = im2col(input, kernel_shape)
  kernel_matrix = im2col(kernel, kernel_shape).transpose(0, 1)
  print("input:")
  print(input_matrix)
  print("kernel matrix:")
  print(kernel_matrix)


'''
@brief: generate the folder name for each layer
@examples:
  - features.0.0.weight -> feature_0_0
  - classifier.1._packed_params._packed_params -> classifier_1
'''
def generate_folder_name(name: str) -> str:
  name_list = name.split(".")
  last_name = name_list[-1]
  if(last_name == "weight"):
    folder_name = "_".join(name_list[:-1])
  elif(last_name == "_packed_params"):
    folder_name = "_".join(name_list[:-2])
  return folder_name

def is_weight_name(name: str) -> bool:
  last_name = name.split(".")[-1]
  if(last_name == "weight" or last_name == "_packed_params"):
    return True
  return False

def dump_weights(model, output_path):
  for layer_name in model.state_dict():
    if(is_weight_name(layer_name)):
      print(layer_name)
      last_name = layer_name.split(".")[-1]
      if(last_name == "weight"):
        weight = model.state_dict()[layer_name].int_repr()
        kernel_size = (weight.shape[2], weight.shape[3])
        weight_matrix = im2col(weight, kernel_size)
      else:
        weight = model.state_dict()[layer_name][0].int_repr()
        weight_matrix = weight.transpose(0, 1)
      folder_name = generate_folder_name(layer_name)
      folder_path = os.path.join(output_path, folder_name)
      if not os.path.exists(folder_path):
        os.makedirs(folder_path)
      weight_path = os.path.join(folder_path, "weight_{}_{}.txt".format(
        weight_matrix.shape[0], weight_matrix.shape[1]))
      np.savetxt(weight_path, weight_matrix.numpy(), fmt="% 4i") # right align


def test():
  a = np.array([[1, 2, 3], [4, 5, 6]])
  np.savetxt("test.txt", a, fmt="% 4i")

def main():
  model_path = "./output/quantized-models/quant_mobilenet_v2.pth"
  output_path = "./output/params/mobilenet_v2"
  # resnet50(quantize=True), mobilenet_v2(quantize=True), qvgg16.vgg16()
  model = mobilenet_v2(quantize=True) 
  model = load_quantized_model(model_path, model, self_quantized=False)
  dump_weights(model, output_path)

if __name__ == "__main__":
  main()