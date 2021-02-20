#include "buff.hpp"

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

namespace {

const double kARMOR_WIDTH = 125.;
const double kSMALL_ARMOR_LENGTH = 135.;
const double KBIG_ARMOR_LENGTH = 230.;
const double kARMOR_HEIGHT = kARMOR_WIDTH * std::sin(75. / 180. * M_PI);
const double kARMOR_DEPTH = kARMOR_WIDTH * std::cos(75. / 180. * M_PI);
const double kHIT_DEPTH = kARMOR_WIDTH / 2. * std::cos(75. / 180. * M_PI);

const cv::Rect kSMALL_ARMOR_FACE(0, 0, kSMALL_ARMOR_LENGTH, kARMOR_WIDTH);
const cv::Rect kBIG_ARMOR_FACE(0, 0, KBIG_ARMOR_LENGTH, kARMOR_WIDTH);

const cv::Matx43d kCOORD_BIG_ARMOR(-KBIG_ARMOR_LENGTH / 2., -kARMOR_HEIGHT / 2,
                                   kARMOR_DEPTH, KBIG_ARMOR_LENGTH / 2.,
                                   -kARMOR_HEIGHT / 2, 0.,
                                   KBIG_ARMOR_LENGTH / 2., kARMOR_HEIGHT / 2,
                                   0., -KBIG_ARMOR_LENGTH / 2.,
                                   kARMOR_HEIGHT / 2, kARMOR_DEPTH);

const cv ::Point3f kHIT_TARGET(0., 0., kHIT_DEPTH);

}  // namespace

Buff::Buff() { SPDLOG_TRACE("Constructed."); }

Buff::Buff(std::vector<std::vector<cv::Point2f>> contours) {
  contours_ = contours;
  SPDLOG_TRACE("Constructed.");
}

Buff::~Buff() { SPDLOG_TRACE("Destructed."); }

void Buff::Init() { SPDLOG_DEBUG("Inited."); }

bool IsTarget() {
  bool target = false;
  // TODO:
  // SVM(contours_)
  return target;
}

std::vector<cv::Point2f> Buff::Vertices2D(cv::RotatedRect rect) {
  std::vector<cv::Point2f> vertices(4);
  rect.points(vertices.data());
  return vertices;
}

std::vector<Armor> Buff::GetArmors() {
  SPDLOG_DEBUG("armors_: {}", armors_.size());
  return armors_;
}

void Buff::SetArmors(std::vector<Armor> armors) { armors_ = armors; }

std::vector<std::vector<cv::Point2f>> Buff::GetContours() {
  SPDLOG_DEBUG("contours_: {}", contours_.size());
  return contours_;
}

void Buff::SetContours(std::vector<std::vector<cv::Point2f>> contours) {
  contours_ = contours;
}

std::vector<cv::RotatedRect> Buff::GetTracks() {
  SPDLOG_DEBUG("rects_: {}", tracks_.size());
  return tracks_;
}

void Buff::SetTracks(std::vector<cv::RotatedRect> tracks) { tracks_ = tracks; }

game::Team Armor::GetTeam() {
  SPDLOG_DEBUG("team_: {}", team_);
  return team_;
}

void Armor::SetTeam(game::Team team) { team_ = team; }
