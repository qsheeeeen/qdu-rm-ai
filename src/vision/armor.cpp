#include "armor.hpp"

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

namespace {

const double kSMALL_ARMOR_WIDTH = 135.;
const double KBIG_ARMOR_WIDTH = 230.;
const double kARMOR_HEIGHT = 125. * std::sin(75. / 180. * M_PI);
const double kARMOR_DEPTH = 125. * std::cos(75. / 180. * M_PI);
const double kHIT_DEPTH = 125. / 2. * std::cos(75. / 180. * M_PI);

const cv::Matx43d kCOORD_SMALL_ARMOR(
    -kSMALL_ARMOR_WIDTH / 2., -kARMOR_HEIGHT / 2, kARMOR_DEPTH,
    kSMALL_ARMOR_WIDTH / 2., -kARMOR_HEIGHT / 2, 0.,
    kSMALL_ARMOR_WIDTH / 2., kARMOR_HEIGHT / 2, 0.,
    -kSMALL_ARMOR_WIDTH / 2., kARMOR_HEIGHT / 2, kARMOR_DEPTH);

const cv::Matx43d kCOORD_BIG_ARMOR(
    -KBIG_ARMOR_WIDTH / 2., -kARMOR_HEIGHT / 2, kARMOR_DEPTH,
    KBIG_ARMOR_WIDTH / 2., -kARMOR_HEIGHT / 2, 0.,
    KBIG_ARMOR_WIDTH / 2., kARMOR_HEIGHT / 2, 0.,
    -KBIG_ARMOR_WIDTH / 2., kARMOR_HEIGHT / 2, kARMOR_DEPTH);

const cv ::Point3f kHIT_TARGET(0., 0., kHIT_DEPTH);

}  // namespace

void Armor::FormRect() {
  cv::Point2f center = (left_bar_.Center() + right_bar_.Center()) / 2.;
  double width = cv::norm(left_bar_.Center() - right_bar_.Center());
  double height = (left_bar_.Length() + right_bar_.Length());

  rect_ = cv::RotatedRect(center, cv::Size(width, height),
                          (left_bar_.Angle() + right_bar_.Angle()) / 2.);

  SPDLOG_DEBUG("center: ({}, {})", center.x, center.y);
  SPDLOG_DEBUG("width, height:  ({}, {})", width, height);
}

Armor::Armor() { SPDLOG_TRACE("Constructed."); }

Armor::Armor(const LightBar &left_bar, const LightBar &right_bar) {
  Init(left_bar, right_bar);
  SPDLOG_TRACE("Constructed.");
}

Armor::~Armor() { SPDLOG_TRACE("Destructed."); }

void Armor::Init(const LightBar &left_bar, const LightBar &right_bar) {
  left_bar_ = left_bar;
  right_bar_ = right_bar;

  FormRect();
  SPDLOG_DEBUG("Inited.");
}

game::Team Armor::GetTeam() {
  SPDLOG_DEBUG("team_: {}", team_);
  return team_;
}

void Armor::SetTeam(game::Team team) { team_ = team; }

game::Model Armor::GetModel() {
  SPDLOG_DEBUG("model_: {}", model_);
  return model_;
}

void Armor::SetModel(game::Model model) { model_ = model; }

const cv::Point2f &Armor::Center2D() {
  SPDLOG_DEBUG("rect_.center: ({}, {})", rect_.center.x, rect_.center.y);
  return rect_.center;
}

std::vector<cv::Point2f> Armor::Vertices2D() {
  std::vector<cv::Point2f> vertices(4);
  rect_.points(vertices.data());
  return vertices;
}

double Armor::Angle2D() {
  SPDLOG_DEBUG("rect_.angle: {}", rect_.angle);
  return rect_.angle;
}

cv::Mat Armor::Face2D(const cv::Mat &frame) {
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

const cv::Mat &Armor::GetRotVec() { return rot_vec_; }

void Armor::SetRotVec(const cv::Mat &rot_vec) {
  rot_vec_ = rot_vec;
  cv::Rodrigues(rot_vec_, rot_mat_);
}

const cv::Mat &Armor::GetRotMat() { return rot_mat_; }

void Armor::SetRotMat(const cv::Mat &rot_mat) {
  rot_mat_ = rot_mat;
  cv::Rodrigues(rot_mat_, rot_vec_);
}

cv::Mat &Armor::GetTransVec() { return trans_vec_; }

void Armor::SetTransVec(const cv::Mat &trans_vec) { trans_vec_ = trans_vec; }

cv::Vec3d Armor::RotationAxis() {
  cv::Vec3d axis(rot_mat_.at<double>(2, 1) - rot_mat_.at<double>(1, 2),
                 rot_mat_.at<double>(0, 2) - rot_mat_.at<double>(2, 0),
                 rot_mat_.at<double>(1, 0) - rot_mat_.at<double>(0, 1));
  return axis;
}

const cv::Mat Armor::Vertices3D() {
  if (GetModel() == game::Model::kHERO ||
      GetModel() == game::Model::kINFANTRY) {
    return cv::Mat(kCOORD_BIG_ARMOR);
  } else {
    return cv::Mat(kCOORD_SMALL_ARMOR);
  }
}

cv::Point3f Armor::HitTarget() {
  auto point_mat = cv::Mat(kHIT_TARGET).reshape(1).t();
  cv::Point3f target(cv::Mat(point_mat * rot_mat_ + trans_vec_));
  return target;
}

const cv::Point3f Armor::WorldCoord() {
  // world_coord_ =  

}
