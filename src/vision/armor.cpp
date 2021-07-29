#include "armor.hpp"

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

namespace {

const double kARMOR_WIDTH = 125.;
const double kARMOR_LENGTH_SMALL = 135.;
const double kARMOR_LENGTH_BIG = 230.;
const double kARMOR_HEIGHT = kARMOR_WIDTH * std::sin(75. / 180. * M_PI);
const double kARMOR_DEPTH = kARMOR_WIDTH * std::cos(75. / 180. * M_PI);
const double kHIT_DEPTH = kARMOR_WIDTH / 2. * std::cos(75. / 180. * M_PI);

std::vector<cv::Point2f> kDST_POV_SMALL{
    cv::Point(0, kARMOR_WIDTH),
    cv::Point(0, 0),
    cv::Point(kARMOR_LENGTH_SMALL, 0),
    cv::Point(kARMOR_LENGTH_SMALL, kARMOR_WIDTH),
};

std::vector<cv::Point2f> kDST_POV_BIG{
    cv::Point(0, kARMOR_WIDTH),
    cv::Point(0, 0),
    cv::Point(kARMOR_LENGTH_BIG, 0),
    cv::Point(kARMOR_LENGTH_BIG, kARMOR_WIDTH),
};

/* clang-format off */
const cv::Matx43d kCOORD_SMALL_ARMOR(
    -kARMOR_LENGTH_SMALL / 2., -kARMOR_HEIGHT / 2, kARMOR_DEPTH,
    kARMOR_LENGTH_SMALL / 2., -kARMOR_HEIGHT / 2, 0.,
    kARMOR_LENGTH_SMALL / 2., kARMOR_HEIGHT / 2, 0.,
    -kARMOR_LENGTH_SMALL / 2., kARMOR_HEIGHT / 2, kARMOR_DEPTH);

const cv::Matx43d kCOORD_BIG_ARMOR(
    -kARMOR_LENGTH_BIG / 2., -kARMOR_HEIGHT / 2, kARMOR_DEPTH,
    kARMOR_LENGTH_BIG / 2., -kARMOR_HEIGHT / 2, 0.,
    kARMOR_LENGTH_BIG / 2., kARMOR_HEIGHT / 2, 0.,
    -kARMOR_LENGTH_BIG / 2., kARMOR_HEIGHT / 2, kARMOR_DEPTH);

/* clang-format on */

const cv::Point3f kHIT_TARGET(0., 0., kHIT_DEPTH);

}  // namespace

/**************/
/* ImageObject */
/* 对于图像世界Armor的抽象 */

cv::RotatedRect ImageObject::FormRect(const LightBar &left_bar,
                                      const LightBar &right_bar) {
  const cv::Point2f center = (left_bar.Center() + right_bar.Center()) / 2.;
  const cv::Size size(cv::norm(left_bar.Center() - right_bar.Center()),
                      (left_bar.Length() + right_bar.Length()));
  const float angle = (left_bar.Angle() + right_bar.Angle()) / 2.f;
  return cv::RotatedRect(center, size, angle);
}

ImageObject::ImageObject() { SPDLOG_TRACE("Constructed."); }

ImageObject::ImageObject(const LightBar &left_bar, const LightBar &right_bar) {
  rect_ = FormRect(left_bar, right_bar);
  SPDLOG_TRACE("Constructed.");
}

ImageObject::ImageObject(const cv::RotatedRect &rect) {
  rect_ = rect;
  SPDLOG_TRACE("Constructed.");
}

ImageObject::~ImageObject() { SPDLOG_TRACE("Destructed."); }

const cv::RotatedRect &ImageObject::GetRect() const { return rect_; }

void ImageObject::SetRect(const cv::RotatedRect &rect) { rect_ = rect; }

const cv::Point2f &ImageObject::SurfaceCenter() const {
  SPDLOG_DEBUG("rect_.center: ({}, {})", rect_.center.x, rect_.center.y);
  return rect_.center;
}

std::vector<cv::Point2f> ImageObject::SurfaceVertices() const {
  std::vector<cv::Point2f> vertices(4);
  rect_.points(vertices.data());
  return vertices;
}

double ImageObject::SurfaceAngle() const {
  SPDLOG_DEBUG("rect_.angle: {}", rect_.angle);
  return rect_.angle;
}

cv::Mat ImageObject::Face(const cv::Mat &frame) const {
  cv::Mat p, t;
  double len;
  if (AspectRatio() > 1.2) {
    t = cv::getPerspectiveTransform(SurfaceVertices(), kDST_POV_BIG);
    len = kARMOR_LENGTH_BIG;
  } else {
    t = cv::getPerspectiveTransform(SurfaceVertices(), kDST_POV_SMALL);
    len = kARMOR_LENGTH_SMALL;
  }
  cv::warpPerspective(frame, p, t, cv::Size(len, kARMOR_WIDTH));

  const int offset_w = (p.cols - kARMOR_WIDTH) / 2;
  const int offset_h = (p.rows - kARMOR_WIDTH) / 2;
  p = p(cv::Rect(offset_w, offset_h, kARMOR_WIDTH, kARMOR_WIDTH));
  cv::cvtColor(p, p, cv::COLOR_RGB2GRAY);
  cv::medianBlur(p, p, 1);
#if 0
  cv::equalizeHist(p, p); /* Tried. No help. */
#endif
  cv::threshold(p, p, 0., 255., cv::THRESH_BINARY | cv::THRESH_TRIANGLE);
  return p;
}

double ImageObject::AspectRatio() const {
  double aspect_ratio = std::max(rect_.size.height, rect_.size.width) /
                        std::min(rect_.size.height, rect_.size.width);
  SPDLOG_DEBUG("aspect_ratio: {}", aspect_ratio);
  return aspect_ratio;
}

/***************/

/***************/
/* PhysicObject */
/* 对于现实世界的Armor的抽象 */

PhysicObject::PhysicObject() { SPDLOG_TRACE("Constructed."); }

PhysicObject::~PhysicObject() { SPDLOG_TRACE("Destructed."); }

game::Model PhysicObject::GetModel() const {
  SPDLOG_DEBUG("model_: {}", model_);
  return model_;
}

void PhysicObject::SetModel(game::Model model) { model_ = model; }

const cv::Mat &PhysicObject::GetRotVec() const { return rot_vec_; }

void PhysicObject::SetRotVec(const cv::Mat &rot_vec) {
  rot_vec_ = rot_vec;
  cv::Rodrigues(rot_vec_, rot_mat_);
}

const cv::Mat &PhysicObject::GetRotMat() const { return rot_mat_; }

void PhysicObject::SetRotMat(const cv::Mat &rot_mat) {
  rot_mat_ = rot_mat;
  cv::Rodrigues(rot_mat_, rot_vec_);
}

const cv::Mat &PhysicObject::GetTransVec() const { return trans_vec_; }

void PhysicObject::SetTransVec(const cv::Mat &trans_vec) {
  trans_vec_ = trans_vec;
}

cv::Vec3d PhysicObject::RotationAxis() const {
  cv::Vec3d axis(rot_mat_.at<double>(2, 1) - rot_mat_.at<double>(1, 2),
                 rot_mat_.at<double>(0, 2) - rot_mat_.at<double>(2, 0),
                 rot_mat_.at<double>(1, 0) - rot_mat_.at<double>(0, 1));
  return axis;
}

const cv::Mat PhysicObject::ModelVertices() const {
  if (game::HasBigArmor(GetModel())) {
    return cv::Mat(kCOORD_BIG_ARMOR);
  } else {
    return cv::Mat(kCOORD_SMALL_ARMOR);
  }
}

/*******************/

/*********/
/* Armor */

Armor::Armor() { SPDLOG_TRACE("Constructed."); }

Armor::Armor(const LightBar &left_bar, const LightBar &right_bar)
    : ImageObject(left_bar, right_bar) {
  SPDLOG_TRACE("Constructed.");
}

Armor::Armor(const cv::RotatedRect &rect) {
  SetRect(rect);
  SPDLOG_TRACE("Constructed.");
}

Armor::~Armor() { SPDLOG_TRACE("Destructed."); }

game::Team Armor::GetTeam() const {
  SPDLOG_DEBUG("team_: {}", team_);
  return team_;
}

void Armor::SetTeam(game::Team team) { team_ = team; }

component::Euler Armor::GetAimEuler() const { return aiming_euler_; }

void Armor::SetAimEuler(const component::Euler &elur) { aiming_euler_ = elur; }

/*********/