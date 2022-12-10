// Some useful functions for model evaluation
#ifndef _UTILS_H
#define _UTILS_H

#include <string>
#include <vector>
#include <torch/torch.h>
#include <torch/script.h>
using std::string;
using std::vector;

class AverageMeter
{
private:
  string name;
  size_t val, sum, count;
  float avg;
public:
  float GetAvg(){ return avg; }
  void update(size_t val, size_t n);
  AverageMeter(string name);
  ~AverageMeter(){}
};

vector<float> GetAcc(torch::Tensor &output, torch::Tensor &target, vector<size_t>
  topk = {1});
float Evaluate(torch::jit::script::Module model, string data_path, size_t batch_size,
	size_t neval_batches);
#endif