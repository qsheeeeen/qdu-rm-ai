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

  game::Team GetTeam();
  game::Model GetModel();

  cv::Point3f Center3D();
  std::vector<cv::Point3f> Vertices3D();
  cv::Mat GetRotMat();
  cv::Vec3d RotationAxis();
  cv::Mat Translation();
};
