#include "utils.h"
#include "dataset.h"
#include <algorithm>
#include <iomanip>


AverageMeter::AverageMeter(string name){
	this->name = name;
	this->avg = 0;
	this->sum = 0;
	this->count = 0;
	this->val = 0;
}

void AverageMeter::update(float val, size_t n){
	this->val = val;
	this->sum += val * n;
	this->count += n;
	this->avg = this->sum / this->count;
}


// Computes the accuracy over the k top predictions for the specified values of k
vector<float> GetAcc(torch::Tensor &output, torch::Tensor &target, vector<size_t> 
	topk){
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

// Evaluate the script model
float Evaluate(torch::jit::script::Module model, string data_path, size_t batch_size,
	size_t neval_batches){
	std::cout << "====== Begin Evaluation ======" << std::endl;
	model.eval();
	auto top1 = AverageMeter(static_cast<string>("top 1"));
	vector<double> norm_mean = {0.485, 0.456, 0.406};
	vector<double> norm_std = {0.229, 0.224, 0.225};
	size_t cnt = 0;
	auto mdataset = myDataset(data_path).map(torch::data::transforms::Normalize<>(norm_mean, norm_std))
		.map(torch::data::transforms::Stack<>());
	auto mdataloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>
		(std::move(mdataset), batch_size);
	for(auto &batch: *mdataloader){
		std::vector<torch::jit::IValue> inputs;
		auto data = batch.data;
		auto target = batch.target;
		inputs.push_back(data);
		auto output = model.forward(inputs).toTensor();
		auto acc1 = GetAcc(output, target)[0];
		std::cout << std::fixed;
		std::cout << "batch: " << cnt << ", acc1: " << setprecision(2) << acc1 << std::endl;
		top1.update(acc1, batch_size);
		cnt++;
		if(cnt >= neval_batches) break;
	}
	return top1.GetAvg();
}