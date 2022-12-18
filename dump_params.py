'''
@brief: dump all the parameters for conv and fc layers.
'''
import torch
import torch.nn as nn
import numpy as np
import os
import re
import qvgg16
from quantization import load_quantized_model
from torchvision.models.quantization import resnet50, ResNet50_QuantizedWeights
from torchvision.models.quantization import mobilenet_v2, MobileNet_V2_QuantizedWeights
from utils import FeatureExtractor, load_imagenet

'''
@input: 
	- tensor data, 4 dimensions: (N, C, H, W)
	- kernel size, 2 dimensions: (H, W)
@output:
	- 2 dimensions: (N * OH, OW)
'''
def im2col(data: torch.Tensor, kernel_size: tuple, dtype=torch.int8, dilation=1, 
	padding=0, stride=1):
	data = data.to(torch.float32)
	unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=padding, 
		stride=stride)
	result = unfold(data).permute(0, 2, 1)
	width = result.shape[2]
	result = result.reshape([-1, width]).to(dtype)
	return result


def im2col_test():
	input = torch.rand([2, 3, 3, 3])
	kernel = torch.rand([2, 3, 2, 2])
	kernel_shape = (kernel.shape[2], kernel.shape[3])
	input_matrix = im2col(input, kernel_shape, dtype=torch.float32, padding=1, stride=2)
	kernel_matrix = im2col(kernel, kernel_shape,dtype=torch.float32, stride=2).transpose(0, 1)
	print("input matrix size: ", input_matrix.shape)
	print("kernel matrix size: ", kernel_matrix.shape)


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

'''The params should be a 2D tensor'''
def save_params(param: torch.tensor, output_path: str, full_name: str, param_type="weight"
	) -> None:
	folder_name = generate_folder_name(full_name)
	folder_path = os.path.join(output_path, folder_name)
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)
	param_path = os.path.join(folder_path, "{}_{}_{}.txt".format(param_type,
		param.shape[0], param.shape[1]))
	np.savetxt(param_path, param.numpy(), fmt="% 4i") # right align

'''For the conv layers whose groups are more than 1'''
def save_group_params(folder_path: str, groups: int) -> None:
	files = os.listdir(folder_path)
	for file_name in files:
		if(re.match(r"(weight|input)_(?!(group)).*.txt", file_name)):
			file_path = os.path.join(folder_path, file_name)
			data = np.loadtxt(file_path, dtype=int)
			height = data.shape[0]
			data = data.reshape((groups, height, -1), order = 'F')
			for group_id in range(groups):
				param_name = file_name.split("_", 1)[0]
				param = data[group_id]
				param_path = os.path.join(folder_path, "{}_group{}_{}_{}.txt".format(param_name,
					group_id, param.shape[0], param.shape[1]))
				np.savetxt(param_path, data[group_id], fmt="% 4i")
			os.remove(file_path)


def find_zero_point(data: torch.Tensor, config: str) -> torch.Tensor:
	if(config == "fbgemm"):
		zero_point = data.q_per_channel_zero_points()
		if (len(data.shape) == 4):
			zero_point = zero_point.unsqueeze(1).unsqueeze(1).unsqueeze(1)
		elif (len(data.shape) == 2):
			zero_point = zero_point.unsqueeze(1)
	elif(config == "qnnpack"):
		zero_point = data.q_zero_point()
	return zero_point


def dump_weights(model: nn.Module, output_path: str, config: str):
	for full_name in model.state_dict():
		if(is_weight_name(full_name)):
			print(full_name)
			last_name = full_name.split(".")[-1]
			if(last_name == "weight"):
				weight = model.state_dict()[full_name]
				zero_point = find_zero_point(weight, config)
				weight = (weight.int_repr() - zero_point).to(torch.int8)
				kernel_size = (weight.shape[2], weight.shape[3])
				weight_matrix = im2col(weight, kernel_size).transpose(0, 1)
			else:
				weight = model.state_dict()[full_name][0]
				zero_point = find_zero_point(weight, config)
				weight = (weight.int_repr() - zero_point).to(torch.int8)
				weight_matrix = weight.transpose(0, 1)
			save_params(weight_matrix, output_path, full_name)


'''Only dump one batch features'''
def dump_inputs(model: nn.Module, output_path: str, batch_size = 20):
	data_path = "~/code/dataset/imagenet"
	_, testloader = load_imagenet(data_path, batch_size)
	inputs, _ = next(iter(testloader))
	layer_names = []
	for full_name in model.state_dict():
		if(is_weight_name(full_name)):
			folder_name = generate_folder_name(full_name)
			layer_name = ".".join(folder_name.split("_"))
			layer_names.append(layer_name)
	model_features = FeatureExtractor(model, layer_names)
	in_features, _ = model_features(inputs)
	in_matrix_features = {}
	# name = "classifier.0"
	for full_name in model.state_dict():
		if(is_weight_name(full_name)):
			last_name = full_name.split(".")[-1]
			folder_name = generate_folder_name(full_name)
			layer_name = ".".join(folder_name.split("_"))
			print(layer_name)
			if(last_name == "weight"): # conv layers
				layer = dict([*model.named_modules()])[layer_name]
				in_features[layer_name] = (in_features[layer_name].int_repr() - \
					in_features[layer_name].q_zero_point()).to(torch.int8)
				in_matrix_features[layer_name] = im2col(in_features[layer_name], kernel_size=
					layer.kernel_size, dilation=layer.dilation, padding=layer.padding, stride=
					layer.stride)
				save_params(in_matrix_features[layer_name], output_path, full_name, 
					param_type="input")
				if(layer.groups > 1):
					folder_path = os.path.join(output_path, folder_name)
					save_group_params(folder_path, layer.groups)
			else: # fc layers
				in_features[layer_name] = (in_features[layer_name].int_repr() - \
					in_features[layer_name].q_zero_point()).to(torch.int8)
				in_matrix_features[layer_name] = in_features[layer_name]
				save_params(in_matrix_features[layer_name], output_path, full_name, 
					param_type="input")

"""Delete the specific .txt file in the folder path"""
def delete_txts(param_type="weight", folder_path="./output/params") -> None:
	if(param_type == "weight"):
		pattern = r'weight.*\.txt'
	else:
		pattern = r'input.*\.txt'
	for root, dirs, files in os.walk(folder_path):
		for file_name in files:
			match_res = re.match(pattern, file_name)
			file_path = os.path.join(root, file_name)
			if(match_res):
				print(file_path)
				os.remove(file_path)


def test():
	save_group_params("./output/test", 2)

def main():
	model_name = "vgg16"
	model_path = "./output/quantized-models/quant_{}.pth".format(model_name)
	output_path = "./output/params/" + model_name
	model_dict = {"vgg16": qvgg16.vgg16(), "resnet50": resnet50(quantize=True), 
		"mobilenet_v2": mobilenet_v2(quantize=True)}
	self_quantized_dict = {"vgg16": True, "resnet50": False, "mobilenet_v2": False}
	quantized_configs = {"vgg16": "fbgemm", "resnet50": "fbgemm", "mobilenet_v2": "qnnpack"}
	model = model_dict[model_name]
	model = load_quantized_model(model_path, model, self_quantized_dict[model_name])
	# dump_weights(model, output_path, quantized_configs[model_name])
	dump_inputs(model, output_path)

if __name__ == "__main__":
	main()