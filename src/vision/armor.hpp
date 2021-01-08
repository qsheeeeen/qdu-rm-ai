#pragma once

#include <vector>

#include "game.hpp"
#include "light_bar.hpp"
#include "opencv2/opencv.hpp"

class Armor {
 private:
  LighjtBar left_bar_, right_bar_;
  cv::RotatedRect rect_;
  game::Team team_ = game::Team::kUNKNOWN;
  game::Model model_ = game::Model::kUNKNOWN;
  cv::Mat face_;

  void FormRect();
  void DetectTeam();

 public:
  Armor();
  Armor(const LighjtBar &left_bar, const LighjtBar &right_bar);
  ~Armor();

  void Init(const LighjtBar &left_bar, const LighjtBar &right_bar);

  game::Team Team(const cv::Mat &frame);
  game::Model GetModel();
  void SetModel(game::Model model);
  const cv::Point2f &Center();
  float Angle();
  cv::Mat Face(const cv::Mat &frame);
};
