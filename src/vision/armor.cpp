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

cv::RotatedRect Armor::FormRect(const LightBar &left_bar,
                                const LightBar &right_bar) {
  const cv::Point2f center = (left_bar.Center() + right_bar.Center()) / 2.;
  const cv::Size size(cv::norm(left_bar.Center() - right_bar.Center()),
                      (left_bar.Length() + right_bar.Length()));
  const float angle = (left_bar.Angle() + right_bar.Angle()) / 2.f;
  return cv::RotatedRect(center, size, angle);
}

void Armor::Init() {
  image_center_ = rect_.center;
  image_angle_ = rect_.angle;

  image_vertices_.resize(4);
  rect_.points(image_vertices_.data());

  double len;
  if (AspectRatio() > 1.2) {
    trans_ = cv::getPerspectiveTransform(SurfaceVertices(), kDST_POV_BIG);
    len = kARMOR_LENGTH_BIG;
  } else {
    trans_ = cv::getPerspectiveTransform(SurfaceVertices(), kDST_POV_SMALL);
    len = kARMOR_LENGTH_SMALL;
  }
  face_size_ = cv::Size(len, kARMOR_WIDTH);
}

Armor::Armor() { SPDLOG_TRACE("Constructed."); }

Armor::Armor(const LightBar &left_bar, const LightBar &right_bar) {
  rect_ = FormRect(left_bar, right_bar);
  Init();
  SPDLOG_TRACE("Constructed.");
}

Armor::Armor(const cv::RotatedRect &rect) {
  rect_ = rect;
  Init();
  SPDLOG_TRACE("Constructed.");
}

Armor::~Armor() { SPDLOG_TRACE("Destructed."); }

double Armor::AspectRatio() const {
  double aspect_ratio = std::max(rect_.size.height, rect_.size.width) /
                        std::min(rect_.size.height, rect_.size.width);
  return aspect_ratio;
}

std::vector<cv::Point2f> Armor::SurfaceVertices() const {
  return image_vertices_;
}

game::Model Armor::GetModel() const { return model_; }
void Armor::SetModel(game::Model model) {
  model_ = model;

  if (game::HasBigArmor(model_)) {
    vertices_ = cv::Mat(kCOORD_BIG_ARMOR);
  } else {
    vertices_ = cv::Mat(kCOORD_SMALL_ARMOR);
  }
}

const cv::RotatedRect &Armor::GetRect() const { return rect_; }
void Armor::SetRect(const cv::RotatedRect &rect) { rect_ = rect; }

game::Team Armor::GetTeam() const { return team_; }
void Armor::SetTeam(game::Team team) { team_ = team; }

component::Euler Armor::GetAimEuler() const { return aiming_euler_; }
void Armor::SetAimEuler(const component::Euler &elur) { aiming_euler_ = elur; }
