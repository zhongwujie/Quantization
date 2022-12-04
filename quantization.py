'''
@dataset: imagenet
@reference: https://github.com/Forggtensky/Quantize_Pytorch_Vgg16AndMobileNet
'''
import torch
import torch.utils.data
import time
from torch import nn
from tqdm import tqdm
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models.quantization import resnet50, ResNet50_QuantizedWeights
from torchvision.models.quantization import mobilenet_v2, MobileNet_V2_QuantizedWeights

from utils import *
import qvgg16


''' evaluate the model, using imagenet '''
def evaluate_model(data_path, batch_size, num_eval_batches, criterion, model):
  print("====== begin evaluating ======")
  model.eval()
  trainloader, testloader = load_imagenet(data_path, batch_size)
  print("num_eval_batches:{}, batch_size:{}".format(num_eval_batches, batch_size))
  with torch.no_grad():
    top1, top5 = evaluate(model, criterion, testloader, neval_batches = num_eval_batches)
    print('number of image %d :Evaluation accuracy: %2.2f'%(batch_size * num_eval_batches, 
      top1.avg))

''' evaluate models in the output'''
def evaluate_saved_model(data_path, batch_size, num_eval_batches, criterion):
  model = qvgg16.vgg16()
  file_path = "./output/script-models/quant_vgg16.pt"
  load_model(file_path, model)
  evaluate_model(data_path, batch_size, num_eval_batches, criterion, model)

''' evaluate the script models'''
def evaluate_script_model(data_path, batch_size, num_eval_batches, criterion):
  file_path = "./output/script-models/quant_resnet50.pt"
  model = torch.jit.load(file_path)
  evaluate_model(data_path, batch_size, num_eval_batches, criterion, model)


def save_quantized_models():
  model1 = resnet50(weights=ResNet50_QuantizedWeights.IMAGENET1K_FBGEMM_V2, quantize=True)
  model2 = mobilenet_v2(weights=MobileNet_V2_QuantizedWeights.IMAGENET1K_QNNPACK_V1, 
    quantize=True)
  torch.save(model1.state_dict(), "./quantized-models/quant_resnet50.pth")
  torch.save(model2.state_dict(), "./quantized-models/quant_mobilenet_v2.pth")

def save_quantized_params(quantized_model):
	with open("./params_int.txt",'w') as f:
		for layer,param in quantized_model.state_dict().items(): # param is weight or bias(Tensor)         
			print(layer, type(param))
			f.write(layer)
			f.write('\n')
			f.write(str(param.int_repr()))
			break


def QAT(data_path, batch_size, num_eval_batches, criterion):
  model =  qvgg16.vgg16()
  qat_model = load_model("./output/float-models/float_vgg16.pth", model)
  trainloader, testloader = load_imagenet(data_path, batch_size)
  optimizer = torch.optim.SGD(qat_model.parameters(), lr = 0.0001)
  qat_model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
  torch.quantization.prepare_qat(qat_model, inplace=True)
  # QAT takes time and one needs to train over a few epochs.
  # Train and check accuracy after each epoch
  print("====== begin training ======")
  for nepoch in range(8):
    print("nepoch: ", nepoch)
    train_one_epoch(qat_model, criterion, optimizer, trainloader, 
        torch.device('cpu'), 1)
    if nepoch > 3:
        # Freeze quantizer parameters
        qat_model.apply(torch.quantization.disable_observer)
    if nepoch > 2:
        # Freeze batch norm mean and variance estimates
        qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
  print("====== end training ======")
  quantized_model = torch.quantization.convert(qat_model.eval(), inplace=False)
  evaluate_model(data_path, batch_size, num_eval_batches, criterion, quantized_model)
  return quantized_model


'''convert a model to script model, can only be used for vgg16'''
def convert_model(src_path, dest_path, model, self_quantized):
  if self_quantized:
    model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    quantized_model = torch.quantization.convert(model.eval(), inplace=True)
  else:
    quantized_model = model
  load_model(src_path, quantized_model)
  sm = torch.jit.script(quantized_model)
  sm.save(dest_path)

'''Convert all the quantized models to script models'''
def convert_models():
  src_folder = "./output/quantized-models"
  dest_folder = "./output/script-models"
  models = [resnet50(quantize=True), mobilenet_v2(quantize=True), qvgg16.vgg16()]
  model_names = ["quant_resnet50.pth", "quant_mobilenet_v2.pth", "quant_vgg16.pth"]
  self_quantize_sigs = [False, False, True]
  for i in range(len(models)):
    src_path = os.path.join(src_folder, model_names[i])
    output_name = model_names[i].split(".")[0] + ".pt"
    dest_path = os.path.join(dest_folder, output_name)
    print(dest_path)
    convert_model(src_path, dest_path, models[i], self_quantize_sigs[i])


def main():
  data_path = "~/code/dataset/imagenet"
  batch_size = 250
  num_eval_batches = 20
  criterion = nn.CrossEntropyLoss()
  evaluate_script_model(data_path, batch_size, num_eval_batches, criterion)
  # torch.save(model.state_dict(), "./output/quantized-models/quant_vgg16.pth")


if __name__ == "__main__":
  main()