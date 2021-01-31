#include "math.hpp"
#include "range_estimator.hpp"

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

namespace {

const double kG = 9.8;                // g = 9.8 m/s²
const double kHEIGHT_ARMOR = 0.10213; // h_armor = 0.10213 m
const double kHEIGHT_SHELL = 0.34503; // h_shell = 0.34503 m

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

bool RangeEstimator::IsOrthogonal(cv::Mat src)
{
  cv::Mat src_transpose = src.t();
  cv::Mat should_be_identity = src_transpose * src;
  cv::Mat idty = cv::Mat::eye(3, 3, CV_32FC3);
  double n = cv::norm(idty - should_be_identity, cv::NORM_L2);
  if (n < 1e-6)
    return true;
  else
    return false;
}

euler_angle RangeEstimator::Estimate(Armor &armor, double bullet_speed)
{
  // 单位化
  double theta = sqrt(pow(rotations_.back().at<double>(0, 0) , 2) + pow(rotations_.back().at<double>(1, 0), 2) +
                      pow(rotations_.back().at<double>(1, 0), 2));
  cv::Mat r_unitization = rotations_.front() / theta;
  rotations_.pop_back();

  // 取值
  double r00 = r_unitization.at<double>(0, 0);
  double r10 = r_unitization.at<double>(1, 0);
  double r20 = r_unitization.at<double>(2, 0);

  // 构造中间传递阵
  cv::Mat r_transfer(3, 3,CV_32F, cv::Scalar(0));
  r_transfer.push_back((0.0, -r20, r10));
  r_transfer.push_back((r20, 0.0, -r00));
  r_transfer.push_back((-r10, r00, 0.0));

  // 求解旋转矩阵
  cv::Mat identity_matrix = cv::Mat::eye(3, 3, CV_32FC3);
  cv::Mat r_transpose = r_unitization.t();
  cv::Mat r_matrix =
      cos(theta) * identity_matrix + (1 - cos(theta)) * r_unitization * r_transpose + sin(theta) * r_transfer;

  euler_angle target_armor;
  if (IsOrthogonal(r_matrix))
  {
    // 奇异性判定
    double singular = sqrt(pow(r_matrix.at<double>(0, 0), 2) + pow(r_matrix.at<double>(1, 0), 2));

    // 解算目标位置的欧拉角
    if (singular >= 1e-6)
    {
      target_armor.pitch = atan2(r_matrix.at<double>(2, 1), r_matrix.at<double>(2, 2));
      target_armor.yaw = atan2(-r_matrix.at<double>(2, 0), singular);
      target_armor.roll = 0.0;
    }
    else
    {
      target_armor.pitch = atan2(-r_matrix.at<double>(1, 2), r_matrix.at<double>(1, 1));
      target_armor.yaw = atan2(-r_matrix.at<double>(2, 0), singular);
      target_armor.roll = 0.0;
    }

    // 解算需要调整到的欧拉角
    double t = tan(target_armor.pitch);
    double b = bullet_speed * bullet_speed;
    if (target_center.y < 540)
    {
      target_armor.pitch =
          atan((b * t - sqrt(pow(b, 2) * t * t - pow((kG * kHEIGHT_ARMOR), 2) - 2 * kG * kHEIGHT_ARMOR * b * t * t)) / (kG * kHEIGHT_ARMOR));
    }
    else
    {
      target_armor.pitch =
          atan((-2 * b * t + sqrt(4 * pow(b, 2) * t * t - 4 * ((bullet_speed * t) * t + kG * kHEIGHT_SHELL) * (kG * kHEIGHT_SHELL - 2 * (bullet_speed * t) * t))) /
               (2 * ((bullet_speed * t) * t + kG * kHEIGHT_SHELL)));
    }

    // 转化为角度
    target_armor.pitch *= 180.0 / 3.1415926;
    target_armor.yaw *= 180.0 / 3.1415926;
  }
  else
  {
    SPDLOG_ERROR("Orthogonal Error.");
  }
  return target_armor;
}

void RangeEstimator::VisualizeResult(cv::Mat& output, bool add_lable) {
  std::vector<cv::Point3f> nose_end_point3D{cv::Point3f(0., 0., 500.0)};
  std::vector<cv::Point2f> nose_end_point2D;

  // for (const auto& rotation : rotations_) {
  //   cv::projectPoints(nose_end_point3D, rotation, translation, cam_mat_,
  //                     distor_coff_, nose_end_point2D);
  // }
}