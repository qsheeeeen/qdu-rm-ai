#pragma once

#include "armor.hpp"

class Compensator {
 private:
  double ballet_speed_;
  cv::Mat cam_mat_, distor_coff_;

  cv::Point3f EstimateWorldCoord(Armor& armor);
  double PinHoleEstimate(Armor& armor);
  double SolveSurfaceLanchAngle(cv::Point2f target);

  void VisualizePnp(Armor& armor, const cv::Mat& output, bool add_lable);

 public:
  Compensator();
  Compensator(const std::string& path);
  ~Compensator();

  void LoadCameraMat(const std::string& path);

  void Apply(std::vector<Armor>& armors, const cv::Mat &frame);

  void VisualizeResult(std::vector<Armor>& armors, const cv::Mat& output,
                       int verbose = 1);
};
