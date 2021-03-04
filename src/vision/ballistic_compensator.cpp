#include "ballistic_compensator.hpp"

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

namespace {

double kG = 9.8;

}  // namespace

BallisticCompensator::BallisticCompensator() { SPDLOG_TRACE("Constructed."); }

BallisticCompensator::~BallisticCompensator() { SPDLOG_TRACE("Destructed."); }

void BallisticCompensator::LoadCameraMat(const std::string& path) {
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

void BallisticCompensator::Estimate3D(Armor& armor) {
  cv::Mat rot_vec, trans_vec;
  cv::solvePnP(armor.Vertices3D(), armor.Vertices2D(), cam_mat_, distor_coff_,
               rot_vec, trans_vec, false, cv::SOLVEPNP_IPPE);
  armor.SetRotVec(rot_vec), armor.SetTransVec(trans_vec);
}

/**
 * @brief Angle θ required to hit coordinate (x, y)
 *
 * {\displaystyle \tan \theta ={\left({\frac {v^{2}\pm {\sqrt
 * {v^{4}-g(gx^{2}+2yv^{2})}}}{gx}}\right)}}
 *
 * @param x coordinate x
 * @param y coordinate y
 * @return double angle
 */
double BallisticCompensator::SolveLanchAngle(double x, double y) {
  const double v_2 = pow(ballet_speed_, 2);
  const double up_base = std::sqrt(std::pow(ballet_speed_, 4) -
                                   kG * (kG * std::pow(x, 2) + 2 * y * v_2));
  const double low = kG * x;
  const double ans1 = std::atan2(v_2 + up_base, low);
  const double ans2 = std::atan2(v_2 - up_base, low);

  if (std::isnan(ans1)) return std::isnan(ans2) ? 0. : ans2;
  if (std::isnan(ans2)) return std::isnan(ans1) ? 0. : ans1;
  return std::min(ans1, ans2);
}
