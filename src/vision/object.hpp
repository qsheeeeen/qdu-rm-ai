#pragma once

#include <vector>

#include "common.hpp"
#include "light_bar.hpp"
#include "opencv2/opencv.hpp"

class ImageObject {
 private:
  cv::RotatedRect rect_;

  cv::RotatedRect FormRect(const LightBar &left_bar, const LightBar &right_bar);

 public:
  ImageObject();
  ImageObject(const cv::RotatedRect &rect);
  ImageObject(const LightBar &left_bar, const LightBar &right_bar);
  ~ImageObject();

  const cv::RotatedRect &GetRect() const;
  void SetRect(const cv::RotatedRect &rect);

  const cv::Point2f &SurfaceCenter() const;
  std::vector<cv::Point2f> SurfaceVertices() const;
  double SurfaceAngle() const;
  cv::Mat Face(const cv::Mat &frame) const;
  double AspectRatio() const;
};

class PhysicObject {
 private:
  cv::Mat rot_vec_, rot_mat_, trans_vec_;
  game::Model model_ = game::Model::kUNKNOWN;

 public:
  PhysicObject();
  ~PhysicObject();

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
