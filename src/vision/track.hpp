#include <cmath>

#include "opencv2/opencv.hpp"

namespace rotation {

enum class Direction { kUNKNOWN, kCLOCKWISE, kANTI };

std::string DirectionToString(rotation::Direction direction);

}  // namespace rotation

namespace cal {

/**
 * @brief 比赛规则旋转速度公式
 *
 * @param temp 输入值
 * @param flag 计算方式，为真则正运算，为假则逆运算
 */
double Speed(double temp, bool flag);
double DeltaTheta(double t, double kDELTA);
double Dist(cv::Point2f a, cv::Point2f b);
double Angle(cv::Point2f a, cv::Point2f center);

}  // namespace cal
