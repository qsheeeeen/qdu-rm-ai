#include "armor_classifier.hpp"

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

ArmorClassifier::ArmorClassifier() {}
ArmorClassifier::~ArmorClassifier() {}

void ArmorClassifier::StoreModel(const std::string& path) {}
void ArmorClassifier::LoadModel(const std::string& path) {}

void ArmorClassifier::Train() {}

void ArmorClassifier::ClassifyModel(Armor &armor) {
  armor.Model() = game::Model::kINFANTRY;
}

void ArmorClassifier::ClassifyTeam(Armor &armor) {
  armor.Team() = game::Team::kBLUE;
}