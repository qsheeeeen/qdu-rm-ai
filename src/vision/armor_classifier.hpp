#pragma once

#include <vector>

#include "armor.hpp"
#include "common.hpp"
#include "opencv2/ml.hpp"

class ArmorClassifier {
 private:
  cv::Ptr<cv::ml::SVM> svm_;

 public:
  ArmorClassifier();
  ~ArmorClassifier();

  void StoreModel(const std::string &path);
  void LoadModel(const std::string &path);

  void Train();
  void ClassifyModel(Armor &armor, const cv::Mat &frame);
  void ClassifyTeam(Armor &armor, const cv::Mat &frame);
};
