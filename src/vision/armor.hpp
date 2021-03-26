#pragma once

#include <vector>

#include "common.hpp"
#include "light_bar.hpp"
#include "opencv2/opencv.hpp"

class Armor {
 private:
  cv::RotatedRect rect_;
  cv::Mat face_, rot_vec_, rot_mat_, trans_vec_;

  game::Team team_ = game::Team::kUNKNOWN;
  game::Model model_ = game::Model::kUNKNOWN;

  LightBar left_bar_, right_bar_;

  common::Euler aiming_euler_;

  cv::RotatedRect FormRect(const LightBar &left_bar, const LightBar &right_bar);

 public:
  Armor();
  Armor(const LightBar &left_bar, const LightBar &right_bar);
  Armor(const cv::RotatedRect &rect);
  ~Armor();

  game::Team GetTeam() const;
  void SetTeam(game::Team team);

  game::Model GetModel() const;
  void SetModel(game::Model model);

  const cv::Mat &GetRotVec() const;
  void SetRotVec(const cv::Mat &rot_vec);

  const cv::Mat &GetRotMat() const;
  void SetRotMat(const cv::Mat &rot_mat);

  const cv::Mat &GetTransVec() const;
  void SetTransVec(const cv::Mat &trans_vec);

  common::Euler GetAimEuler() const;
  void SetAimEuler(const common::Euler &elur);

  const cv::Point2f &SurfaceCenter() const;
  std::vector<cv::Point2f> SurfaceVertices() const;
  double SurfaceAngle() const;
  cv::Mat Face(const cv::Mat &frame) const;
  double AspectRatio() const;

  cv::Vec3d RotationAxis() const;
  const cv::Mat ModelVertices() const;
};
