#pragma once

#include "armor.hpp"

class Compensator {
 private:
  double ballet_speed_;
  cv::Mat cam_mat_, distor_coff_;

  void Estimate3D(Armor& armor);
  double PinHoleEstimate(Armor& armor);
  double SolveSurfaceLanchAngle(cv::Point2d target);

  void VisualizeEstimate3D(const cv::Mat& output, int verbose);

 public:
  Compensator();
  ~Compensator();

  void LoadCameraMat(const std::string& path);

  void VisualizeResult(const cv::Mat& output, int verbose = 1);
};
