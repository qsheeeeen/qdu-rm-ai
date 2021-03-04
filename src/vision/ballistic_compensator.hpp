#pragma once

#include "armor.hpp"

class BallisticCompensator {
 private:
  double ballet_speed_;
  cv::Mat cam_mat_, distor_coff_;

 public:
  BallisticCompensator(/* args */);
  ~BallisticCompensator();

  void LoadCameraMat(const std::string& path);
  void Estimate3D(Armor& armor);
  /**
   * @brief Angle θ required to hit coordinate (x, y)
   *
   * {\displaystyle \tan \theta ={\left({\frac {v^{2}\pm {\sqrt
   * {v^{4}-g(gx^{2}+2yv^{2})}}}{gx}}\right)}}
   *
   * @param x coordinate x
   * @param y coordinate y
   * @return double angle
   */
  double SolveLanchAngle(double x, double y);
};