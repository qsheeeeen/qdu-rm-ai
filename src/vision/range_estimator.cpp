#include "range_estimator.hpp"

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

namespace {

const double kSMALL_ARMOR_WIDTH = 135.;
const double KBIG_ARMOR_WIDTH = 230.;
const double kARMOR_HEIGHT = 120.74072829;
const double kARMOR_DEPTH = 32.35238064;

const std::vector<cv::Point3f> kCOORD_SMALL_ARMOR{
    cv::Point3f(-kSMALL_ARMOR_WIDTH / 2., kARMOR_HEIGHT / 2, kARMOR_DEPTH),
    cv::Point3f(kSMALL_ARMOR_WIDTH / 2., kARMOR_HEIGHT / 2, 0.),
    cv::Point3f(kSMALL_ARMOR_WIDTH / 2., kARMOR_HEIGHT / 2, 0.),
    cv::Point3f(-kSMALL_ARMOR_WIDTH / 2., kARMOR_HEIGHT / 2, kARMOR_DEPTH)};

const std::vector<cv::Point3f> kCOORD_BIG_ARMOR{
    cv::Point3f(-KBIG_ARMOR_WIDTH / 2., kARMOR_HEIGHT / 2, kARMOR_DEPTH),
    cv::Point3f(KBIG_ARMOR_WIDTH / 2., kARMOR_HEIGHT / 2, 0.),
    cv::Point3f(KBIG_ARMOR_WIDTH / 2., kARMOR_HEIGHT / 2, 0.),
    cv::Point3f(-KBIG_ARMOR_WIDTH / 2., kARMOR_HEIGHT / 2, kARMOR_DEPTH)};

}  // namespace

void RangeEstimator::LoadCameraMat(const std::string& path) {
  fs_.open(path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);

  if (fs_.isOpened()) {
    cam_mat_ = fs_[cam_model_]["cam_mat"].mat();
    distor_coff_ = fs_[cam_model_]["distor_coff"].mat();
    SPDLOG_DEBUG("Loaded cali data.");
  } else {
    SPDLOG_ERROR("Can not load cali data for '{}' in '{}'", cam_model_, path);
  }
  fs_.release();
}

void RangeEstimator::PnpEstimate(Armor& armor) {
  cv::Mat rotation, translation;
  if (armor.GetModel() == game::Model::kHERO ||
      armor.GetModel() == game::Model::kINFANTRY) {
    cv::solvePnP(kCOORD_BIG_ARMOR, armor.Vertices2D(), cam_mat_, distor_coff_,
                 rotation, translation, false, cv::SOLVEPNP_IPPE);
  } else {
    cv::solvePnP(kCOORD_SMALL_ARMOR, armor.Vertices2D(), cam_mat_, distor_coff_,
                 rotation, translation, false, cv::SOLVEPNP_IPPE);
  }
  rotations_.push_back(rotation);
  translations_.push_back(translation);
}

double RangeEstimator::PinHoleEstimate(std::vector<cv::Point2f> target) {}

RangeEstimator::RangeEstimator() { SPDLOG_TRACE("Constructed."); }

RangeEstimator::RangeEstimator(const std::string& cam_model) {
  Init(cam_model);
  SPDLOG_TRACE("Constructed.");
}

RangeEstimator::~RangeEstimator() { SPDLOG_TRACE("Destructed."); }

void RangeEstimator::Init(const std::string& cam_model) { SPDLOG_DEBUG("Inited."); }

double RangeEstimator::Estimate(Armor& armor, double bullet_speed) {}

void RangeEstimator::VisualizeResult(cv::Mat& output, bool add_lable) {
  std::vector<cv::Point3f> nose_end_point3D{cv::Point3f(0., 0., 500.0)};
  std::vector<cv::Point2f> nose_end_point2D;

  // for (const auto& rotation : rotations_) {
  //   cv::projectPoints(nose_end_point3D, rotation, translation, cam_mat_,
  //                     distor_coff_, nose_end_point2D);
  // }
}