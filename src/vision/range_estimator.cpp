#include "range_estimator.hpp"

#include <math.h>

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

namespace {

const double kG = 9.8;
const double kHEIGHT_ARMOR = 0.10213;
const double kHEIGHT_SHELL = 0.34503;

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
  cv::Mat rot_vec, trans_vec;
  cv::solvePnP(armor.SolidVertices(), armor.SurfaceVertices(), cam_mat_, distor_coff_,
               rot_vec, trans_vec, false, cv::SOLVEPNP_IPPE);
  rotations_.push_back(rot_vec);
  translations_.push_back(trans_vec);
}

double RangeEstimator::PinHoleEstimate(std::vector<cv::Point2f> target) {
  double fx = cam_mat_.at<double>(0, 0);
  double fy = cam_mat_.at<double>(1, 1);
  double cx = cam_mat_.at<double>(0, 2);
  double cy = cam_mat_.at<double>(1, 2);
  cv::Point2f pnt;

  std::vector<cv::Point2f> out;
  target.push_back(target_center);

  //对像素点去畸变
  cv::undistortPoints(target, out, cam_mat_, distor_coff_, cv::noArray(),
                      cam_mat_);
  pnt = out.front();

  //去畸变后的比值
  double rxNew = (pnt.x - cx) / fx;
  double ryNew = (pnt.y - cy) / fy;

  euler_angle_.yaw = atan(rxNew) / CV_PI * 180;
  euler_angle_.pitch = -atan(ryNew) / CV_PI * 180;
}

RangeEstimator::RangeEstimator() { SPDLOG_TRACE("Constructed."); }

RangeEstimator::RangeEstimator(const std::string& cam_model) {
  Init(cam_model);
  SPDLOG_TRACE("Constructed.");
}

RangeEstimator::~RangeEstimator() { SPDLOG_TRACE("Destructed."); }

void RangeEstimator::Init(const std::string& cam_model) {
  SPDLOG_DEBUG("Inited.");
}

bool RangeEstimator::IsOrthogonal(cv::Mat src) {
  cv::Mat should_be_identity = src.t() * src;
  cv::Mat idty = cv::Mat::eye(3, 3, CV_32FC3);
  double n = cv::norm(idty - should_be_identity, cv::NORM_L2);
  if (n < 1e-6)
    return true;
  else
    return false;
}

int RangeEstimator::Estimate(Armor& armor, double bullet_speed) {
  cv::Mat r_matrix(3, 3, CV_32F, cv::Scalar(0));
  cv::Rodrigues(armor.GetRotVec(), r_matrix, cv::noArray());

  if (IsOrthogonal(r_matrix)) {
    // 奇异性判定
    double singular = sqrt(pow(r_matrix.at<double>(0, 0), 2) +
                           pow(r_matrix.at<double>(1, 0), 2));

    // 解算目标位置的欧拉角
    if (singular >= 1e-6) {
      euler_angle_.pitch =
          atan2(r_matrix.at<double>(2, 1), r_matrix.at<double>(2, 2));
    } else {
      euler_angle_.pitch =
          atan2(-r_matrix.at<double>(1, 2), r_matrix.at<double>(1, 1));
    }
    euler_angle_.yaw = atan2(-r_matrix.at<double>(2, 0), singular);
    euler_angle_.roll = 0.0;
    // 解算需要调整到的欧拉角
    double t = tan(euler_angle_.pitch);
    double b = bullet_speed * bullet_speed;
    if (target_center.y < 540) {
      euler_angle_.pitch =
          atan((b * t - sqrt(pow(b, 2) * t * t - pow((kG * kHEIGHT_ARMOR), 2) -
                             2 * kG * kHEIGHT_ARMOR * b * t * t)) /
               (kG * kHEIGHT_ARMOR));
    } else {
      euler_angle_.pitch =
          atan((-2 * b * t +
                sqrt(4 * pow(b, 2) * t * t -
                     4 * ((bullet_speed * t) * t + kG * kHEIGHT_SHELL) *
                         (kG * kHEIGHT_SHELL - 2 * (bullet_speed * t) * t))) /
               (2 * ((bullet_speed * t) * t + kG * kHEIGHT_SHELL)));
    }

    // 转化为角度
    euler_angle_.pitch *= 180.0 / CV_PI;
    euler_angle_.yaw *= 180.0 / CV_PI;
  } else {
    SPDLOG_ERROR("Orthogonal Error.");
  }
  return 0;
}

void RangeEstimator::VisualizeResult(cv::Mat& output, bool add_lable) {
  std::vector<cv::Point3f> nose_end_point3D{cv::Point3f(0., 0., 500.0)};
  std::vector<cv::Point2f> nose_end_point2D;

  // for (const auto& rotation : rotations_) {
  //   cv::projectPoints(nose_end_point3D, rotation, translation, cam_mat_,
  //                     distor_coff_, nose_end_point2D);
  // }
}