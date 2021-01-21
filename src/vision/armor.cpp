#include "armor.hpp"

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

void Armor::FormRect() {
  cv::Point2f center = (left_bar_.Center() + right_bar_.Center()) / 2.;
  double width = cv::norm(left_bar_.Center() - right_bar_.Center());
  double height = (left_bar_.Length() + right_bar_.Length());

  rect_ = cv::RotatedRect(center, cv::Size(width, height),
                          (left_bar_.Angle() + right_bar_.Angle()) / 2.);

  SPDLOG_DEBUG("center: ({}, {})", center.x, center.y);
  SPDLOG_DEBUG("width, height:  ({}, {})", width, height);
}

Armor::Armor() { SPDLOG_DEBUG("Constructed."); }

Armor::Armor(const LightBar &left_bar, const LightBar &right_bar) {
  Init(left_bar, right_bar);
  SPDLOG_DEBUG("Constructed.");
}

Armor::~Armor() { SPDLOG_DEBUG("Destructed."); }

void Armor::Init(const LightBar &left_bar, const LightBar &right_bar) {
  left_bar_ = left_bar;
  right_bar_ = right_bar;

  FormRect();
  SPDLOG_DEBUG("Inited.");
}

game::Team &Armor::Team() {
  SPDLOG_DEBUG("team_: {}", team_);
  return team_;
}

game::Model &Armor::Model() { return model_; }

const cv::Point2f &Armor::Center() {
  SPDLOG_DEBUG("rect_.center: ({}, {})", rect_.center.x, rect_.center.y);
  return rect_.center;
}

std::vector<cv::Point2f> Armor::Vertices() {
  std::vector<cv::Point2f> vertices(4);
  rect_.points(vertices.data());
  return vertices;
}

double Armor::Angle() {
  SPDLOG_DEBUG("rect_.angle: {}", rect_.angle);
  return rect_.angle;
}

cv::Mat Armor::Face(const cv::Mat &frame) {
  cv::Point2f src_vertices[4];
  rect_.points(src_vertices);

  cv::Point2f dst_vertices[4];
  cv::Rect dst_rect(0, 0, 100, 100);  // TODO
  dst_vertices[0] = dst_rect.tl() + cv::Point(0, dst_rect.height);
  dst_vertices[1] = dst_rect.tl();
  dst_vertices[2] = dst_rect.br() - cv::Point(0, dst_rect.height);
  dst_vertices[3] = dst_rect.br();

  cv::Mat trans_mat = cv::getPerspectiveTransform(src_vertices, dst_vertices);

  cv::Mat perspective;
  cv::warpPerspective(frame, perspective, trans_mat, dst_rect.size());
  cv::adaptiveThreshold(perspective, perspective, 255,
                        cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 3, 1);
  return perspective;
}

cv::Mat &Armor::Rotation() { return rotation_; }

cv::Vec3d Armor::RotationAxis() {
  cv::Vec3d axis(rotation_.at<double>(2, 1) - rotation_.at<double>(1, 2),
                 rotation_.at<double>(0, 2) - rotation_.at<double>(2, 0),
                 rotation_.at<double>(1, 0) - rotation_.at<double>(0, 1));
  return axis;
}

cv::Mat &Armor::Translation() { return translation_; }