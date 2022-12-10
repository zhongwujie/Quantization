#include <torch/torch.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include "utils.h"
#include "dataset.h"
using std::string;

void CVTest(){
  string img_path = "../pictures/lena.png";
  cv::Mat image;
  image = cv::imread(img_path, 1);
  cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
  cv::imshow("Display Image", image);
  cv::waitKey(0);
}

void ModelTest(){
  vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({1, 3, 224, 224}));
  string model_path = "../../output/script-models/quant_vgg16.pt";
  torch::jit::script::Module model = torch::jit::load(model_path);
  at::Tensor output = model.forward(inputs).toTensor();
  std::cout << "output: " << output << std::endl;
}

void Run(){
  string data_path = "/home/zhong/code/dataset/imagenet/ILSVRC/Data/CLS-LOC/val_sorted";
  string model_path = "../../output/script-models/quant_vgg16.pt";
  torch::jit::script::Module model = torch::jit::load(model_path);
  int batch_size = 250;
  int neval_batches = 20;
  Evaluate(model, data_path, batch_size, neval_batches);
}

int main() {
  Run();
}