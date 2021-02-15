#pragma once

#include <chrono>
#include <vector>

#include "armor_detector.hpp"
#include "buff.hpp"
#include "detector.hpp"
#include "opencv2/opencv.hpp"

struct BuffDetectorParam {
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

class BuffDetector : private Detector<Buff, BuffDetectorParam> {
 private:
  Buff buff_;
  std::vector<Armor> armors_;

  std::chrono::milliseconds duration_armors_, duration_contours_,
      duration_track_;

  void InitDefaultParams(const std::string &path);
  bool PrepareParams(const std::string &path);

  //cv::Mat FrameProcessing(const cv::Mat &frame);
  void FindArmors(const cv::Mat &frame);
  void FindContours(const cv::Mat &frame);
  void FindTrack(const cv::Mat &frame);
  void MatchArmors(const cv::Mat &frame);

  void VisualizeContour(const cv::Mat &output, bool add_lable);
  void VisualizeArmor(const cv::Mat &output, bool add_lable);
  void VisualizeTrack(const cv::Mat &output, bool add_lable);

 public:
  BuffDetector();
  BuffDetector(const std::string &param_path);
  ~BuffDetector();

  const std::vector<Buff> &Detect(const cv::Mat &frame);

  void VisualizeResult(const cv::Mat &frame, int verbose);
};