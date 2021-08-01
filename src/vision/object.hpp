#pragma once

#include <vector>

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

class ImageObject {
 public:
  std::vector<cv::Point2f> image_vertices_;
  cv::Point2f image_center_;
  cv::Size face_size_;
  cv::Mat trans_, face_;
  float image_angle_;

  const cv::Point2f &SurfaceCenter() const { return image_center_; }

  virtual std::vector<cv::Point2f> SurfaceVertices() const = 0;

  double SurfaceAngle() const { return image_angle_; }

  cv::Mat Face(const cv::Mat &frame) const {
    cv::Mat face;
    cv::warpPerspective(frame, face, trans_, face_size_);
    cv::cvtColor(face, face, cv::COLOR_RGB2GRAY);
    cv::medianBlur(face, face, 1);
#if 0
  cv::equalizeHist(face, face); /* Tried. No help. */
#endif
    cv::threshold(face, face, 0., 255.,
                  cv::THRESH_BINARY | cv::THRESH_TRIANGLE);

    /* 截取中间正方形 */
    float min_edge = std::min(face.cols, face.rows);
    const int offset_w = (face.cols - min_edge) / 2;
    const int offset_h = (face.rows - min_edge) / 2;
    face = face(cv::Rect(offset_w, offset_h, min_edge, min_edge));
    return face;
  }
};

class PhysicObject {
 public:
  cv::Mat rot_vec_, rot_mat_, trans_vec_, vertices_;

  const cv::Mat &GetRotVec() const { return rot_vec_; }
  void SetRotVec(const cv::Mat &rot_vec) {
    rot_vec_ = rot_vec;
    cv::Rodrigues(rot_vec_, rot_mat_);
  }

  const cv::Mat &GetRotMat() const { return rot_mat_; }
  void SetRotMat(const cv::Mat &rot_mat) {
    rot_mat_ = rot_mat;
    cv::Rodrigues(rot_mat_, rot_vec_);
  }

  const cv::Mat &GetTransVec() const { return trans_vec_; }
  void SetTransVec(const cv::Mat &trans_vec) { trans_vec_ = trans_vec; }

  cv::Vec3d RotationAxis() const {
    cv::Vec3d axis(rot_mat_.at<double>(2, 1) - rot_mat_.at<double>(1, 2),
                   rot_mat_.at<double>(0, 2) - rot_mat_.at<double>(2, 0),
                   rot_mat_.at<double>(1, 0) - rot_mat_.at<double>(0, 1));
    return axis;
  }
  const cv::Mat ModelVertices() const { return vertices_; }
};
