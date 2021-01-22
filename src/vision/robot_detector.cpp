#include "robot_detector.hpp"

#include <ostream>

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

namespace {

const auto kCV_FONT = cv::FONT_HERSHEY_SIMPLEX;
const auto kGREEN = cv::Scalar(0., 255., 0.);

const double kSMALL_ARMOR_WIDTH = 135.;
const double KBIG_ARMOR_WIDTH = 230.;
const double kARMOR_HEIGHT = 120.74072829;
const double kARMOR_DEPTH = 32.35238064;

const std::vector<cv::Point3f> kCOORD_SMALL_ARMOR{
    cv::Point3f(-kSMALL_ARMOR_WIDTH / 2., -kARMOR_HEIGHT / 2, kARMOR_DEPTH),
    cv::Point3f(kSMALL_ARMOR_WIDTH / 2., -kARMOR_HEIGHT / 2, 0.),
    cv::Point3f(kSMALL_ARMOR_WIDTH / 2., kARMOR_HEIGHT / 2, 0.),
    cv::Point3f(-kSMALL_ARMOR_WIDTH / 2., kARMOR_HEIGHT / 2, kARMOR_DEPTH)};

const std::vector<cv::Point3f> kCOORD_BIG_ARMOR{
    cv::Point3f(-KBIG_ARMOR_WIDTH / 2., -kARMOR_HEIGHT / 2, kARMOR_DEPTH),
    cv::Point3f(KBIG_ARMOR_WIDTH / 2., -kARMOR_HEIGHT / 2, 0.),
    cv::Point3f(KBIG_ARMOR_WIDTH / 2., kARMOR_HEIGHT / 2, 0.),
    cv::Point3f(-KBIG_ARMOR_WIDTH / 2., kARMOR_HEIGHT / 2, kARMOR_DEPTH)};

}  // namespace

void RobotDetector::InitDefaultParams(const std::string& params_path) {
  cv::FileStorage fs(params_path,
                     cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);

  fs << "width_diff_th" << 0.5;
  fs << "height_diff_th" << 0.5;
  fs << "center_dist_low_th" << 1.5;
  fs << "center_dist_high_th" << 5;
  fs << "axis_angle_th" << 0.1;
  SPDLOG_DEBUG("Inited params.");
}

bool RobotDetector::PrepareParams(const std::string& path) {
  cv::FileStorage fs(path,
                     cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);

  if (fs.isOpened()) {
    params_.width_diff_th = fs["width_diff_th"];
    params_.height_diff_th = fs["height_diff_th"];
    params_.center_dist_low_th = fs["center_dist_low_th"];
    params_.center_dist_high_th = fs["center_dist_high_th"];
    params_.axis_angle_th = fs["axis_angle_th"];
    return true;
  } else {
    SPDLOG_ERROR("Can not inited params.");
    return false;
  }
}

void RobotDetector::LoadCameraMat(const std::string& path) {
  cv::FileStorage fs(path,
                     cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);

  if (fs.isOpened()) {
    cam_mat_ = fs[cam_model_]["cam_mat"].mat();
    distor_coff_ = fs[cam_model_]["distor_coff"].mat();
    SPDLOG_DEBUG("Loaded cali data.");
  } else {
    SPDLOG_ERROR("Can not load cali data for '{}' in '{}'", cam_model_, path);
  }
}

void RobotDetector::Estimate3D(Armor &armor) {
  if (armor.Model() == game::Model::kHERO ||
      armor.Model() == game::Model::kINFANTRY) {
    cv::solvePnP(kCOORD_BIG_ARMOR, armor.Vertices(), cam_mat_, distor_coff_,
                 armor.Rotation(), armor.Translation(), false,
                 cv::SOLVEPNP_IPPE);
  } else {
    cv::solvePnP(kCOORD_SMALL_ARMOR, armor.Vertices(), cam_mat_, distor_coff_,
                 armor.Rotation(), armor.Translation(), false,
                 cv::SOLVEPNP_IPPE);
  }
}

double RobotDetector::AxisAngle(cv::Vec3d &axis1, cv::Vec3d &axis2) {
  return std::acos(axis1.dot(axis2) / (cv::norm(axis1) * cv::norm(axis2)));
}

RobotDetector::RobotDetector() { SPDLOG_TRACE("Constructed."); }

RobotDetector::RobotDetector(const std::string& params_path,
                             const std::string& cam_param_path) {
  Init(params_path, cam_param_path);
  SPDLOG_TRACE("Constructed.");
}

RobotDetector::~RobotDetector() { SPDLOG_TRACE("Destructed."); }

void RobotDetector::Init(const std::string& params_path, const std::string& cam_param_path) {
  if (!PrepareParams(params_path)) {
    InitDefaultParams(params_path);
    PrepareParams(params_path);
    SPDLOG_WARN("Can not find parasm file. Created and reloaded.");
  }
  LoadCameraMat(cam_param_path);
  SPDLOG_DEBUG("Inited.");
}

void RobotDetector::Detect(std::vector<Armor> &armors) {
  SPDLOG_DEBUG("Detecting");
  const auto start = std::chrono::high_resolution_clock::now();
  for (auto iti = armors.begin(); iti != armors.end(); ++iti) {
    bool found_match = false;
    Estimate3D(*iti);
    for (auto itj = iti + 1; itj != armors.end(); ++itj) {
      if (iti->Model() != itj->Model()) continue;

      const double height_diff = std::abs(iti->Center().y - itj->Center().y);
      if (height_diff > (params_.height_diff_th * frame_size_.height)) continue;

      const double center_dist = cv::norm(iti->Center() - itj->Center());
      if (center_dist < params_.center_dist_low_th) continue;
      if (center_dist > params_.center_dist_high_th) continue;

      auto axis_i = iti->RotationAxis();
      auto axis_j = itj->RotationAxis();
      if (AxisAngle(axis_i, axis_j) > params_.axis_angle_th) continue;

      found_match = true;
      Estimate3D(*itj);
      robots_.emplace_back(Robot(std::vector<Armor>{*iti, *itj}));
      break;
    }
    if (!found_match) robots_.emplace_back(Robot(*iti));
  }
  SPDLOG_DEBUG("Found robots: {}", robots_.size());

  const auto stop = std::chrono::high_resolution_clock::now();
  duration_robots_ =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  SPDLOG_DEBUG("duration_robots_: {} ms", duration_robots_.count());

  SPDLOG_DEBUG("Detected.");
}

void RobotDetector::VisualizeResult(cv::Mat &output, bool add_lable) {
  std::vector<cv::Point2d> robot_2d;

  if (!robots_.empty()) {
    for (auto &robot : robots_) {
      cv::projectPoints(robot.Vertices(), robot.Rotation(), robot.Translation(),
                        cam_mat_, distor_coff_, robot_2d);

      auto size = robot_2d.size();
      for (size_t i = 0; i < size; ++i) {
        for (size_t j = i + 1; j < size; ++j)
          cv::line(output, robot_2d[i], robot_2d[j], kGREEN);
      }

      if (add_lable) {
        std::ostringstream buf;
        buf << game::TeamToString(robot.Team()) << ", "
            << game::ModelToString(robot.Model());
        cv::putText(output, buf.str(), robot_2d[0], kCV_FONT, 1.0, kGREEN);
      }
    }
  }
}
