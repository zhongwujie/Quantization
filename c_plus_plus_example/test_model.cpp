#include <torch/torch.h>
#include <iostream>
#include <torchvision/vision.h>

int main() {
  torch::Tensor tensor = torch::eye(3);
  std::cout << tensor << std::endl;
}