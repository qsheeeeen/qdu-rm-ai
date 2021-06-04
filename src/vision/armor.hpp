#pragma once

#include <vector>

#include "common.hpp"
#include "light_bar.hpp"
#include "opencv2/opencv.hpp"

class ImageArmor {
 private:
  cv::RotatedRect rect_;

  cv::RotatedRect FormRect(const LightBar &left_bar, const LightBar &right_bar);

 public:
  ImageArmor();
  ImageArmor(const cv::RotatedRect &rect);
  ImageArmor(const LightBar &left_bar, const LightBar &right_bar);
  ~ImageArmor();

  const cv::RotatedRect &GetRect() const;
  void SetRect(const cv::RotatedRect &rect);

  const cv::Point2f &SurfaceCenter() const;
  std::vector<cv::Point2f> SurfaceVertices() const;
  double SurfaceAngle() const;
  cv::Mat Face(const cv::Mat &frame) const;
  double AspectRatio() const;
};

class PhysicArmor {
 private:
  cv::Mat rot_vec_, rot_mat_, trans_vec_;
  game::Model model_ = game::Model::kUNKNOWN;

 public:
  PhysicArmor();
  ~PhysicArmor();

  const cv::Mat &GetRotVec() const;
  void SetRotVec(const cv::Mat &rot_vec);

  const cv::Mat &GetRotMat() const;
  void SetRotMat(const cv::Mat &rot_mat);

  const cv::Mat &GetTransVec() const;
  void SetTransVec(const cv::Mat &trans_vec);

  game::Model GetModel() const;
  void SetModel(game::Model model);

  cv::Vec3d RotationAxis() const;
  const cv::Mat ModelVertices() const;
};

class Armor : public ImageArmor, public PhysicArmor {
 private:
  game::Team team_ = game::Team::kUNKNOWN;
  common::Euler aiming_euler_;

 public:
  Armor();
  Armor(const LightBar &left_bar, const LightBar &right_bar);
  Armor(const cv::RotatedRect &rect);
  ~Armor();

  game::Team GetTeam() const;
  void SetTeam(game::Team team);

  common::Euler GetAimEuler() const;
  void SetAimEuler(const common::Euler &elur);
};
