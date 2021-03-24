#pragma once

#include "armor.hpp"

class Compensator {
 private:
  double ballet_speed_;
  cv::Mat cam_mat_, distor_coff_;

  double PinHoleEstimate(Armor& armor);
  double SolveSurfaceLanchAngle(cv::Point2f target);

  void VisualizePnp(Armor& armor, const cv::Mat& output, bool add_lable);

 public:
  Compensator();
  Compensator(const std::string& cam_mat_path);
  ~Compensator();

  cv::Vec3f EstimateWorldCoord(Armor& armor);

  void LoadCameraMat(const std::string& path);

  void Apply(std::vector<Armor>& armors, const cv::Mat& frame,
             const cv::Mat& rot_mat);

  void VisualizeResult(std::vector<Armor>& armors, const cv::Mat& output,
                       int verbose = 1);
};
