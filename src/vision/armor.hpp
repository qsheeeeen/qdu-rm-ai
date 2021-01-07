#pragma once

#include <vector>

#include "game.hpp"
#include "light_bar.hpp"
#include "opencv2/opencv.hpp"

class Armor {
 private:
  LighjtBar left_bar_, right_bar_;
  cv::Rect rect_;
  game::Team team_;
  game::Model model_;
  cv::Point2f center_;
  cv::Mat face_;

 public:
  Armor();
  Armor(const LighjtBar &left_bar, const LighjtBar &right_bar);
  ~Armor();

  void Init(const LighjtBar &left_bar, const LighjtBar &right_bar);

  const std::vector<cv::Point2f> Vertices();
  const game::Team &Team();
  const game::Model &Model();
  const cv::Point2f &Center();
  float Angle();
  const cv::Mat &Face();
};
