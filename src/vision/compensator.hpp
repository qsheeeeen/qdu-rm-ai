#pragma once

#include "armor.hpp"

class Compensator {
 private:
  double ballet_speed_;
  cv::Mat cam_mat_, distor_coff_;

 public:
  Compensator(/* args */);
  ~Compensator();

  void LoadCameraMat(const std::string& path);
  void Estimate3D(Armor& armor);
  double PinHoleEstimate(Armor& armor);

  /**
   * @brief Angle θ required to hit coordinate (target_x, target_y)
   *
   * {\displaystyle \tan \theta ={\left({\frac {v^{2}\pm {\sqrt
   * {v^{4}-g(gx^{2}+2yv^{2})}}}{gx}}\right)}}
   *
   * @param target 目标坐标
   * @return double 出射角度
   */
  double SolveSurfaceLanchAngle(cv::Point2d target);

  void VisualizeResult(const cv::Mat& output, int verbose = 1);
};