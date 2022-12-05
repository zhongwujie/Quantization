#include "utils.h"
#include <algorithm>

AverageMeter::AverageMeter(string &name){
  this->name = name;
  this->avg = 0;
  this->sum = 0;
  this->count = 0;
  this->val = 0;
}

void AverageMeter::update(size_t val, size_t n){
  this->val = val;
  this->sum += val * n;
  this->count += n;
  this->avg = this->sum / this->count;
}


// Computes the accuracy over the k top predictions for the specified values of k
vector<float> GetAcc(torch::Tensor &output, torch::Tensor &target, vector<size_t> 
  topk = {1}){
  torch::NoGradGuard no_grad;
  auto maxk = *std::max_element(topk.begin(), topk.end());
  auto batch_size = target.size(0);
  
  auto pred = std::get<1>(output.topk(maxk, 1, true, true));
  pred = pred.t();
  auto correct = pred.eq(target.view({1, -1}).expand_as(pred));

  vector<float> res;
  for(auto it = topk.begin(); it < topk.end(); it++){
    auto correct_k = correct.narrow(0, 0, *it).reshape(-1).sum(0, true).item<float>();
    res.push_back(correct_k * 100.0 / batch_size);
  }
  return res;
}

// Evaluate the model
float Evaluate(torch::jit::script::Module module){
}