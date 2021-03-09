#include "compensator.hpp"

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

namespace {

const double kG = 9.80665;
const cv::Scalar kGREEN(0., 255., 0.);
const cv::Scalar kRED(0., 0., 255.);
const cv::Scalar kYELLOW(0., 255., 255.);

}  // namespace

void Compensator::Estimate3D(Armor& armor) {
  cv::Mat rot_vec, trans_vec;
  cv::solvePnP(armor.SolidVertices(), armor.SurfaceVertices(), cam_mat_,
               distor_coff_, rot_vec, trans_vec, false, cv::SOLVEPNP_IPPE);
  armor.SetRotVec(rot_vec), armor.SetTransVec(trans_vec);
  cv::Mat world_coord =
      ((cv::Vec2f(armor.SurfaceCenter()) * cam_mat_.inv() - trans_vec) *
       armor.GetRotMat().inv());
  armor.SetWorldCoord(cv::Point3d(world_coord));
}

/**
 * @brief Angle θ required to hit coordinate (x, y)
 *
 * {\displaystyle \tan \theta ={\left({\frac {v^{2}\pm {\sqrt
 * {v^{4}-g(gx^{2}+2yv^{2})}}}{gx}}\right)}}
 *
 * @param target 目标坐标
 * @return double 出射角度
 */
double Compensator::SolveSurfaceLanchAngle(cv::Point2d target) {
  const double v_2 = pow(ballet_speed_, 2);
  const double up_base =
      std::sqrt(std::pow(ballet_speed_, 4) -
                kG * (kG * std::pow(target.x, 2) + 2 * target.y * v_2));
  const double low = kG * target.x;
  const double ans1 = std::atan2(v_2 + up_base, low);
  const double ans2 = std::atan2(v_2 - up_base, low);

  if (std::isnan(ans1)) return std::isnan(ans2) ? 0. : ans2;
  if (std::isnan(ans2)) return std::isnan(ans1) ? 0. : ans1;
  return std::min(ans1, ans2);
}

void Compensator::VisualizeEstimate3D(const cv::Mat& output, int verbose) {
  std::vector<cv::Point2f> out_points;
  // cv::projectPoints(SurfaceVertices(), rot_vec_, trans_vec_, );

  for (std::size_t i = 0; i < out_points.size(); ++i) {
    cv::line(output, out_points[i], out_points[(i + 1) % out_points.size()],
             kGREEN);
  }
}

Compensator::Compensator() { SPDLOG_TRACE("Constructed."); }

Compensator::~Compensator() { SPDLOG_TRACE("Destructed."); }

void Compensator::LoadCameraMat(const std::string& path) {
  cv::FileStorage fs(path,
                     cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);

  if (fs.isOpened()) {
    cam_mat_ = fs["cam_mat"].mat();
    distor_coff_ = fs["distor_coff"].mat();
    if (cam_mat_.empty() && distor_coff_.empty()) {
      SPDLOG_ERROR("Can not load cali data.");
    } else {
      SPDLOG_DEBUG("Loaded cali data.");
    }
  } else {
    SPDLOG_ERROR("Can not open file: '{}'", path);
  }
}

void Compensator::VisualizeResult(const cv::Mat& output, int verbose) {
  VisualizeEstimate3D(output, verbose);
}
