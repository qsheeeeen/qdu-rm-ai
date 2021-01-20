#pragma once

#include <vector>

#include "armor.hpp"
#include "game.hpp"
#include "opencv2/ml.hpp"

class ArmorClassifier {
 private:
  cv::Ptr<cv::ml::SVM> svm_;

 public:
  ArmorClassifier();
  ~ArmorClassifier();

  void StoreModel(std::string path);
  void LoadModel(std::string path);

  void Train();
  game::Model ClassifyModel(Armor &armor);
  game::Team ClassifyTeam(Armor &armor);
};
