#pragma once

#include <string>

#include "armor.hpp"
#include "game.hpp"
#include "opencv2/opencv.hpp"

class RangeEstimator {
 private:
  cv::FileStorage fs_;
  std::string cam_model_;
  cv::Mat cam_mat_;
  cv::Mat distor_coff_;

  cv::Point2f target_center;

  // AngleSolve target
  struct euler_angle
  {
      double yaw;
      double pitch;
      double roll;
  }euler_angle_;

  std::vector<cv::Mat> rotations_;
  std::vector<cv::Mat> translations_;

  void LoadCameraMat(const std::string& path);
  void PnpEstimate(Armor& armor);
  double PinHoleEstimate(std::vector<cv::Point2f> target);

 public:
  RangeEstimator();
  RangeEstimator(const std::string& cam_model);
  ~RangeEstimator();

  void Init(const std::string& cam_model);
  bool IsOrthogonal(cv::Mat src);
  int Estimate(Armor& armor, double bullet_speed);
  void VisualizeResult(cv::Mat& output, bool add_lable = true);
};
