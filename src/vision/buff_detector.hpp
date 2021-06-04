#pragma once

#include <chrono>
#include <vector>

#include "armor_classifier.hpp"
#include "armor_detector.hpp"
#include "buff.hpp"
#include "detector.hpp"
#include "opencv2/opencv.hpp"

struct BuffDetectorParam {
  double binary_th;
  int se_erosion;    /* erosion in getStructuringElement */
  double ap_erosion; /* erosion in approxPolyDP */

  std::size_t contour_size_low_th;
  double rect_ratio_low_th;
  double rect_ratio_high_th;

  double contour_center_area_low_th;
  double contour_center_area_high_th;
  double rect_center_ratio_low_th;
  double rect_center_ratio_high_th;
};

class BuffDetector : private Detector<Armor, BuffDetectorParam> {
 private:
  Buff buff_;
  std::vector<std::vector<cv::Point>> contours_, contours_poly_;
  std::vector<cv::RotatedRect> rects_;
  std::vector<cv::Point2f> circumference_;
  cv::RotatedRect hammer_;

  std::chrono::milliseconds duration_armors_, duration_center_, duration_rects_;

  void InitDefaultParams(const std::string &path);
  bool PrepareParams(const std::string &path);

  void FindRects(const cv::Mat &frame);

  void MatchArmors();
  void MatchDirection();
  void MatchPredict();

  void VisualizeArmors(const cv::Mat &output, bool add_lable);

 public:
  BuffDetector();
  BuffDetector(const std::string &param_path, game::Team buff_team);
  ~BuffDetector();

  const std::vector<Armor> &Detect(const cv::Mat &frame);

  void VisualizeResult(const cv::Mat &frame, int verbose);
};