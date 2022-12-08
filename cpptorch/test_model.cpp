#include <torch/torch.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include "utils.h"

using std::string;

void CVTest(){
  string img_path = "../pictures/lena.png";
  cv::Mat image;
  image = cv::imread(img_path, 1);
  cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
  cv::imshow("Display Image", image);
  cv::waitKey(0);
}

int main() {
  CVTest();
}