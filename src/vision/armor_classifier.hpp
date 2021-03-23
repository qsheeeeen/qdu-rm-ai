#pragma once

#include <vector>

#include "armor.hpp"
#include "common.hpp"
#include "opencv2/opencv.hpp"

class ArmorClassifier {
 private:
  double conf_;
  int class_id_;
  std::vector<game::Model> classes_;
  cv::dnn::Net net_;
  cv::Size net_input_size_;
  cv::Mat blob_;
  game::Model model_;

 public:
  ArmorClassifier();
  ArmorClassifier(const std::string model_path, const std::string lable_path,
                  const cv::Size &input_size);
  ~ArmorClassifier();

  void LoadModel(const std::string &path);
  void LoadLable(const std::string &path);
  void SetInputSize(const cv::Size &input_size);

  void ClassifyModel(Armor &armor, const cv::Mat &frame);
};
