#include "track.hpp"

namespace rotation {

std::string DirectionToString(rotation::Direction direction) {
  switch (direction) {
    case rotation::Direction::kUNKNOWN:
      return std::string("UNKNOWN");
    case rotation::Direction::kCLOCKWISE:
      return std::string("CLOCKWISE");
    case rotation::Direction::kANTI:
      return std::string("ANTICLOCKWISE");
    default:
      return std::string("UNKNOWN");
  }
}

}  // namespace rotation

namespace cal {

double Speed(double temp, bool flag) {
  if (flag)
    temp = 0.785 * sin(1.884 * temp) + 1.305;
  else
    temp = asin((temp - 1.305) / 1.884) / 0.785;
  return temp;
}

/**
 * $
 * \quad \int^{t_1+\Delta t}_{t_1} 0.785\sin{1.884t}+1.305{\rm d}t \\
 * = 1.305\Delta t+ \dfrac{0.785}{1.884} ( \cos{1.884t} - \cos{1.884(t+\Delta
 * t)}) \\ = \sqrt{2-2\cos{{1.884\Delta t}}}\sin({1.884t} +
 *\arctan{\dfrac{1-\cos{{1.884\Delta t}}}{\sin{{1.884\Delta t}}}}) + 1.305
 *\Delta t
 * $
 */
double DeltaTheta(double t, double kDELTA) {
  // return 1.305 * kDELTA + sqrt(2 - 2 * cos(1.884 * kDELTA)) *
  //                          sin(1.884 * t + atan((1 - cos(1.884 * kDELTA)) /
  //                                             sin(1.884 * kDELTA)));
  return 1.305 * kDELTA +
         0.785 / 1.884 * (cos(1.884 * t) - cos(1.884 * (t + kDELTA)));
}

double Dist(cv::Point2f a, cv::Point2f b) {
  return sqrt(powf(a.x - b.x, 2) + powf(a.y - b.y, 2));
}

double Angle(cv::Point2f a, cv::Point2f center) {
  double angle = atan2(a.y - center.y, a.x - center.x) / CV_PI * 180;
  if (angle < 0) angle += 360;
  return angle;
}

}  // namespace cal
