#pragma once

#include <vector>

#include "game.hpp"
#include "light_bar.hpp"
#include "opencv2/opencv.hpp"

class Armor {
 private:
  LightBar left_bar_, right_bar_;
  cv::RotatedRect rect_;
  game::Team team_ = game::Team::kUNKNOWN;
  game::Model model_ = game::Model::kUNKNOWN;
  cv::Mat face_, rot_vec_, rot_mat_, trans_vec_;

  void FormRect();

 public:
  Armor();
  Armor(const LightBar &left_bar, const LightBar &right_bar);
  ~Armor();

  void Init(const LightBar &left_bar, const LightBar &right_bar);

  game::Team GetTeam();
  void SetTeam(game::Team team);

  game::Model GetModel();
  void SetModel(game::Model model);

  const cv::Point2f &Center2D();
  std::vector<cv::Point2f> Vertices2D();
  double Angle2D();
  cv::Mat Face(const cv::Mat &frame);

  const cv::Mat &GetRotVec();
  void SetRotVec(const cv::Mat &rot_vec);

  const cv::Mat &GetRotMat();
  void SetRotMat(const cv::Mat &rot_mat);

  cv::Mat &GetTransVec();
  void SetTransVec(const cv::Mat &trans_vec);

  cv::Vec3d RotationAxis();
  const std::vector<cv::Point3f> &Vertices3D();
  cv::Point3f HitTarget();
};
