#include "armor.hpp"

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

void Armor::FormRect() {
  cv::Point2f center = (left_bar_.Center() + right_bar_.Center()) / 2.f;
  float width = cv::norm(left_bar_.Center() - right_bar_.Center());
  float height = (left_bar_.Length() + right_bar_.Length()) / 2.f;  // TODO

  rect_ = cv::RotatedRect(center, cv::Size(width, height),
                          (left_bar_.Angle() + right_bar_.Angle()) / 2.f);

  SPDLOG_DEBUG("[Armor] center: ({}, {})", center.x, center.y);
  SPDLOG_DEBUG("[Armor] width, height:  ({}, {})", width, height);
}

void Armor::DetectTeam() {
  team_ = game::Team::kBLUE;
  SPDLOG_DEBUG("[Armor] team_: {}", team_);
}

Armor::Armor() { SPDLOG_DEBUG("[Armor] Constructed."); }

Armor::Armor(const LightBar &left_bar, const LightBar &right_bar)
    : left_bar_(left_bar), right_bar_(right_bar) {
  FormRect();
  SPDLOG_DEBUG("[Armor] Constructed.");
}

Armor::~Armor() { SPDLOG_DEBUG("[Armor] Destructed."); }

void Armor::Init(const LightBar &left_bar, const LightBar &right_bar) {
  left_bar_ = left_bar;
  right_bar_ = right_bar;

  FormRect();
  SPDLOG_DEBUG("[Armor] Inited.");
}

game::Team Armor::Team(const cv::Mat &frame) {
  SPDLOG_DEBUG("[Armor] team_: {}", team_);
  if (team_ == game::Team::kUNKNOWN) DetectTeam();
  return team_;
}

game::Model Armor::GetModel() { return model_; }

void Armor::SetModel(game::Model model) { model_ = model; }

const cv::Point2f &Armor::Center() {
  SPDLOG_DEBUG("[Armor] rect_.center: ({}, {})", rect_.center.x,
               rect_.center.y);
  return rect_.center;
}

float Armor::Angle() {
  SPDLOG_DEBUG("[Armor] rect_.angle: {}", rect_.angle);
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
  return perspective;
}