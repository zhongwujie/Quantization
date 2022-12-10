#pragma once

#include<torch/script.h>
#include<torch/torch.h>
#include<vector>
#include<string>
#include<opencv2/opencv.hpp>

using std::string;
using std::vector;

vector<string> GetEntriesInDirectory(const string& dirPath);
// Traverse the .jpg file in the folder
void load_data_from_folder(string folder_path, vector<string> &image_paths, 
	vector<int> &labels);

class myDataset:public torch::data::Dataset<myDataset>{
public:
	myDataset(string image_dir){
		load_data_from_folder(image_dir, image_paths, labels);
	}
	// Override get() function to return tensor at location index
	torch::data::Example<> get(size_t index) override{
		string image_path = image_paths.at(index);
		cv::Mat image = cv::imread(image_path);
		cv::resize(image, image, cv::Size(224, 224));
		int label = labels.at(index);
		torch::Tensor img_tensor = torch::from_blob(image.data, { image.rows, image.cols, 
			3 }, torch::kByte).permute({ 2, 0, 1 }); // Channels x Height x Width
		img_tensor = img_tensor.to(torch::kF32).div(255);
		torch::Tensor label_tensor = torch::full({ 1 }, label);
		return {img_tensor.clone(), label_tensor.clone()};
	}
	// Return the length of data
	torch::optional<size_t> size() const override {
		return image_paths.size();
	};
private:
	vector<string> image_paths;
	vector<int> labels;
};