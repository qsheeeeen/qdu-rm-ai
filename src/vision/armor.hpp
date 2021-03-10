#pragma once

#include <vector>

#include "game.hpp"
#include "light_bar.hpp"
#include "opencv2/opencv.hpp"

class Armor {
 private:
  cv::RotatedRect rect_;
  cv::Mat face_, rot_vec_, rot_mat_, trans_vec_;

  game::Team team_ = game::Team::kUNKNOWN;
  game::Model model_ = game::Model::kUNKNOWN;

  LightBar left_bar_, right_bar_;

  Euler aiming_euler_;

  void FormRect();

 public:
  Armor();
  Armor(const LightBar &left_bar, const LightBar &right_bar);
  Armor(const cv::RotatedRect &rect);
  ~Armor();

  void Init(const LightBar &left_bar, const LightBar &right_bar);
  void Init(const cv::RotatedRect &rect);

  game::Team GetTeam();
  void SetTeam(game::Team team);

  game::Model GetModel();
  void SetModel(game::Model model);

  const cv::Mat &GetRotVec();
  void SetRotVec(const cv::Mat &rot_vec);

  const cv::Mat &GetRotMat();
  void SetRotMat(const cv::Mat &rot_mat);

  const cv::Mat &GetTransVec();
  void SetTransVec(const cv::Mat &trans_vec);

  Euler GetAimEuler();
  void SetAimEuler(const Euler &elur);
  
  const cv::Point2f &SurfaceCenter();
  std::vector<cv::Point2f> SurfaceVertices();
  double SurfaceAngle();
  cv::Mat Face(const cv::Mat &frame);

  cv::Vec3d RotationAxis();
  const cv::Mat ModelVertices();
};
