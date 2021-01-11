#pragma once

#include <chrono>
#include <vector>

#include "armor.hpp"
#include "armor_classifier.hpp"
#include "game.hpp"
#include "light_bar.hpp"

struct ArmorDetectorParam {
  double binary_th;
  int erosion_size;
  size_t contour_size_th;
  double contour_area_low_th;
  double contour_area_high_th;
  double bar_area_low_th;
  double bar_area_high_th;
  double angle_high_th;
  double aspect_ratio_low_th;
  double aspect_ratio_high_th;
  double angle_diff_th;
  double length_diff_th;
  double height_diff_th;
  double area_diff_th;
  double center_dist_low_th;
  double center_dist_high_th;
};

class ArmorDetector {
 private:
  const cv::Scalar green_ = cv::Scalar(0., 255., 0.);
  cv::Size frame_size_;
  game::Team enemy_team_;
  ArmorClassifier armor_classifier_;
  std::vector<LightBar> lightbars_;
  std::vector<Armor> armors_;

  cv::FileStorage fs_;
  ArmorDetectorParam params_;

  std::chrono::microseconds duration_find_bars_;
  std::chrono::microseconds duration_find_armors_;

  void InitDefaultParams(std::string params_path);
  void PrepareParams();

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

  void Detect(cv::Mat &frame);
  void VisualizeResult(cv::Mat &output, bool draw_bars = false,
                       bool draw_armor = true, bool add_lable = true);
};
