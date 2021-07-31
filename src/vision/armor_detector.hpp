#pragma once

#include <chrono>

#include "armor.hpp"
#include "armor_classifier.hpp"
#include "common.hpp"
#include "detector.hpp"
#include "light_bar.hpp"

struct ArmorDetectorParam {
  double binary_th;
  int se_erosion;    /* erosion in getStructuringElement */
  double ap_erosion; /* erosion in approxPolyDP */
  std::size_t contour_size_low_th;
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

class ArmorDetector : public Detector<Armor, ArmorDetectorParam> {
 private:
  game::Team enemy_team_;
  std::vector<std::vector<cv::Point>> contours_, contours_poly_;
  std::vector<LightBar> lightbars_;

  std::chrono::milliseconds duration_bars_, duration_armors_;

  void InitDefaultParams(const std::string &path);
  bool PrepareParams(const std::string &path);

  void FindLightBars(const cv::Mat &frame);
  void MatchScore(const LightBar &bar1, const LightBar &bar2);
  void MatchLightBars();

  void VisualizeLightBar(const cv::Mat &output, bool add_lable);
  void VisualizeArmor(const cv::Mat &output, bool add_lable);

 public:
  ArmorDetector();
  ArmorDetector(const std::string &params_path, game::Team enemy_team);
  ~ArmorDetector();

  void SetEnemyTeam(game::Team enemy_team);

  const std::vector<Armor> &Detect(const cv::Mat &frame);
  void VisualizeResult(const cv::Mat &output, int verbose = 1);
};
