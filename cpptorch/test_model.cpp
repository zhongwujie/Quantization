#include <torch/torch.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include "utils.h"
#include "dataset.h"
using std::string;

void CVTest(){
  string data_path = "/home/zhong/code/dataset/imagenet/ILSVRC/Data/CLS-LOC/val_sorted";
  string img_path = data_path + "/n01440764/ILSVRC2012_val_00000293.JPEG";
  string model_path = "../../output/script-models/quant_vgg16.pt";
  torch::jit::script::Module model = torch::jit::load(model_path);
  cv::Mat image;
  image = cv::imread(img_path, 1);
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  image.convertTo(image, CV_32FC3);
  image = image / 255;
  cv::resize(image, image, cv::Size(224, 224));
  torch::Tensor img_tensor = torch::from_blob(image.data, { image.rows, image.cols, 
			3 }, torch::kF32).permute({ 2, 0, 1 });
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(img_tensor.unsqueeze(0));
  auto target = model.forward(inputs);
  std::cout << img_tensor[0] << std::endl;
  std::cout << target << std::endl;
}

void ModelTest(){
  vector<torch::jit::IValue> inputs;
  torch::manual_seed(0);
  auto data = torch::rand({1, 3, 224, 224});
  inputs.push_back(data);
  string model_path = "../../output/script-models/quant_vgg16.pt";
  torch::jit::script::Module model = torch::jit::load(model_path);
  at::Tensor output = model.forward(inputs).toTensor();
  std::cout << "data: " << data[0][0] << std::endl;
  std::cout << "output: " << output << std::endl;
}

void Run(){
  string data_path = "/home/zhong/code/dataset/imagenet/ILSVRC/Data/CLS-LOC/val_sorted";
  string model_path = "../../output/script-models/quant_mobilenet_v2.pt";
  torch::jit::script::Module model = torch::jit::load(model_path);
  int batch_size = 250;
  int neval_batches = 20;
  auto acc = Evaluate(model, data_path, batch_size, neval_batches);
  std::cout << "Numer of image: " << batch_size * neval_batches << "  accuracy: "
    << setprecision(2) << acc << std::endl;
}

int main() {
  Run();
}