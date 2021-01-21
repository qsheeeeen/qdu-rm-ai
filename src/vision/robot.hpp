#pragma once

#include <vector>

#include "armor.hpp"
#include "game.hpp"
#include "opencv2/opencv.hpp"

class Robot {
 private:
  std::vector<Armor> armors_;

 public:
  Robot();
  Robot(Armor armor);
  Robot(std::vector<Armor> armors);
  ~Robot();

  void Init(Armor armor);
  void Init(std::vector<Armor> armors);

  game::Team Team();
  game::Model Model();
  cv::Point3f Center();
  std::vector<cv::Point2f> Vertices();
  cv::Mat Rotation();
  cv::Vec3d RotationAxis();
  cv::Mat Translation();
};
