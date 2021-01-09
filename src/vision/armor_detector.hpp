#pragma once

#include <vector>

#include "armor.hpp"
#include "armor_classifier.hpp"
#include "game.hpp"
#include "light_bar.hpp"

class ArmorDetector {
 private:
  const cv::Scalar green_ = cv::Scalar(0., 255., 0.);
  game::Team enemy_team_;
  ArmorClassifier armor_classifier_;
  std::vector<LightBar> lightbars_;
  std::vector<Armor> armors_;

  cv::FileStorage params_;

  void FindLightBars(const cv::Mat &frame);
  void MatchLightBars();
  game::Model GetModel(const Armor &armor);

  void VisualizeLightBar(cv::Mat &output, bool add_lable);
  void VisualizeArmor(cv::Mat &output, bool add_lable);

 public:
  ArmorDetector();
  ArmorDetector(std::string params_path, game::Team enemy_team);
  ~ArmorDetector();

  void Init(std::string params_path, game::Team enemy_team);

  void InitDefaultParams(std::string params_path);

  void Detect(cv::Mat &frame);
  void VisualizeResult(cv::Mat &output, bool draw_bars = false,
                       bool draw_armor = true, bool add_lable = true);
};
