#include "armor_classifier.hpp"

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

ArmorClassifier::ArmorClassifier() {}
ArmorClassifier::~ArmorClassifier() {}

void ArmorClassifier::StoreModel(const std::string& path) {}
void ArmorClassifier::LoadModel(const std::string& path) {}

void ArmorClassifier::Train() {}

void ArmorClassifier::ClassifyModel(Armor &armor, const cv::Mat &frame) {
  armor.SetModel(game::Model::kINFANTRY);
}

void ArmorClassifier::ClassifyTeam(Armor &armor, const cv::Mat &frame) {
  armor.SetTeam(game::Team::kBLUE);
}