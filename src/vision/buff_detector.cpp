#include "buff_detector.hpp"

#include <cmath>

#include "compensator.hpp"
#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

namespace {

const auto kCV_FONT = cv::FONT_HERSHEY_SIMPLEX;
const cv::Scalar kGREEN(0., 255., 0.);
const cv::Scalar kRED(0., 0., 255.);
const cv::Scalar kYELLOW(0., 255., 255.);

const double kDELTA = 0.3;  //总延迟时间

}  // namespace

static double Angle(const cv::Point2f &p, const cv::Point2f &ctr) {
  auto rel = p - ctr;
  return std::atan2(rel.x, rel.y);
}

static double Speed(double temp, bool flag) {
  if (flag)
    temp = 0.785 * std::sin(1.884 * temp) + 1.305;
  else
    temp = std::asin((temp - 1.305) / 1.884) / 0.785;
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
static double DeltaTheta(double t) {
  // return 1.305 * kDELTA + sqrt(2 - 2 * cos(1.884 * kDELTA)) *
  //                          sin(1.884 * t + atan((1 - cos(1.884 * kDELTA)) /
  //                                             sin(1.884 * kDELTA)));
  return 1.305 * kDELTA +
         0.785 / 1.884 * (cos(1.884 * t) - cos(1.884 * (t + kDELTA)));
}

void BuffDetector::InitDefaultParams(const std::string &params_path) {
  cv::FileStorage fs(params_path,
                     cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);

  fs << "binary_th" << 220;
  fs << "se_erosion" << 2;
  fs << "ap_erosion" << 1.;

  fs << "contour_size_low_th" << 2;
  fs << "rect_ratio_low_th" << 0.4;
  fs << "rect_ratio_high_th" << 2.5;

  fs << "contour_center_area_low_th" << 100;
  fs << "contour_center_area_high_th" << 1000;
  fs << "rect_center_ratio_low_th" << 0.6;
  fs << "rect_center_ratio_high_th" << 1.67;
  SPDLOG_DEBUG("Inited params.");
}

bool BuffDetector::PrepareParams(const std::string &params_path) {
  cv::FileStorage fs(params_path,
                     cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
  if (fs.isOpened()) {
    params_.binary_th = fs["binary_th"];
    params_.se_erosion = fs["se_erosion"];
    params_.ap_erosion = fs["ap_erosion"];

    params_.contour_size_low_th = static_cast<int>(fs["contour_size_low_th"]);
    params_.rect_ratio_low_th = fs["rect_ratio_low_th"];
    params_.rect_ratio_high_th = fs["rect_ratio_high_th"];

    params_.contour_center_area_low_th = fs["contour_center_area_low_th"];
    params_.contour_center_area_high_th = fs["contour_center_area_high_th"];
    params_.rect_center_ratio_low_th = fs["rect_center_ratio_low_th"];
    params_.rect_center_ratio_high_th = fs["rect_center_ratio_high_th"];
    return true;
  } else {
    SPDLOG_ERROR("Can not load params.");
    return false;
  }
}

void BuffDetector::FindRects(const cv::Mat &frame) {
  const auto start = high_resolution_clock::now();
  float center_rect_area = params_.contour_center_area_low_th * 1.5;
  rects_.clear();
  hammer_ = cv::RotatedRect();

  frame_size_ = cv::Size(frame.cols, frame.rows);

  cv::Mat channels[3], img;
  cv::split(frame, channels);

#if 1
  if (buff_.GetTeam() == game::Team::kBLUE) {
    img = channels[0] - channels[2];
  } else if (buff_.GetTeam() == game::Team::kRED) {
    img = channels[2] - channels[0];
  }
#else
  if (buff_.GetTeam() == game::Team::kBLUE) {
    result = channels[0] - channels[2];
  } else if (buff_.GetTeam() == game::Team::kRED) {
    result = channels[2] - channels[0];
  }
#endif

  cv::threshold(img, img, params_.binary_th, 255., cv::THRESH_BINARY);

  cv::Mat kernel = cv::getStructuringElement(
      cv::MORPH_RECT,
      cv::Size2i(2 * params_.se_erosion + 1, 2 * params_.se_erosion + 1),
      cv::Point(params_.se_erosion, params_.se_erosion));

  cv::dilate(img, img, kernel);
  cv::morphologyEx(img, img, cv::MORPH_CLOSE, kernel);
  cv::findContours(img, contours_, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

#if 1
  contours_poly_.resize(contours_.size());
  for (size_t i = 0; i < contours_.size(); ++i) {
    cv::approxPolyDP(cv::Mat(contours_[i]), contours_poly_[i],
                     params_.ap_erosion, true);
  }
#endif

  for (const auto &contour : contours_poly_) {
    if (contour.size() < params_.contour_size_low_th) continue;

    cv::RotatedRect rect = cv::minAreaRect(contour);
    double rect_ratio = rect.size.aspectRatio();
    double contour_area = cv::contourArea(contour);
    double rect_area = rect.size.area();

    SPDLOG_DEBUG("contour_area is {}", contour_area);
    SPDLOG_DEBUG("rect_area is {}", rect_area);
    if (contour_area > params_.contour_center_area_low_th &&  // 200 - 500
        contour_area < params_.contour_center_area_high_th) {
      if (rect_ratio < params_.rect_center_ratio_high_th &&  // 0.6 - 1.67
          rect_ratio > params_.rect_center_ratio_low_th) {
        buff_.SetCenter(rect.center);
        center_rect_area = rect_area;
        SPDLOG_WARN("center's area is {}", rect_area);
        continue;
      }
    }

    if (rect_area > 1.2 * contour_area &&  // 它矩形的面积大于1.5倍的轮廓面积
        rect_area > 20 * center_rect_area &&  // 20倍的圆心矩形面积 10000+
        rect_area < 80 * center_rect_area) {  // 80倍的圆心矩形面积 后续拿到json
      hammer_ = rect;
      SPDLOG_DEBUG("hammer_contour's area is {}", contour_area);
      continue;
    }
    if (0 < hammer_.size.area()) {  //宝剑
      if (contour_area > 1.5 * hammer_.size.area()) continue;
      if (rect_area > 0.7 * hammer_.size.area()) continue;
    }

    SPDLOG_DEBUG("rect_ratio is {}", rect_ratio);  // 0.4 - 2.5
    if (rect_ratio < params_.rect_ratio_low_th) continue;
    if (rect_ratio > params_.rect_ratio_high_th) continue;

    if (rect_area < 3 * center_rect_area) continue;  // 3倍的
    if (rect_area > 15 * center_rect_area) continue;

    if (contour_area > rect_area * 1.2) continue;
    if (contour_area < rect_area * 0.8) continue;

    SPDLOG_DEBUG("armor's area is {}", rect_area);

    rects_.emplace_back(rect);
  }

  const auto stop = high_resolution_clock::now();
  duration_rects_ = duration_cast<std::chrono::milliseconds>(stop - start);
}

void BuffDetector::MatchDirection() {
  SPDLOG_DEBUG("start MatchDirection");
  if (buff_.GetDirection() == common::Direction::kUNKNOWN) {
    cv::Point2f center = buff_.GetCenter();
    double angle, sum = 0;
    std::vector<double> angles;

    if (circumference_.size() == 5) {
      for (auto point : circumference_) angle = Angle(point, center);
      angles.emplace_back(angle);
    }

    for (int i = 4; i > 1; i--) {
      double delta = angles[i] - angles[i - 1];
      sum += delta;
    }

    if (sum > 0)
      buff_.SetDirection(common::Direction::kCCW);
    else if (sum == 0)
      buff_.SetDirection(common::Direction::kUNKNOWN);
    else
      buff_.SetDirection(common::Direction::kCW);

    SPDLOG_DEBUG("buff_'s getDirection is {}", buff_.GetDirection());
  }
}

void BuffDetector::MatchArmors() {
  const auto start = high_resolution_clock::now();
  std::vector<Armor> armors;

  for (auto &rect : rects_) {
    armors.emplace_back(Armor(rect));
  }

  SPDLOG_DEBUG("armors.size is {}", armors.size());
  SPDLOG_DEBUG("the buff's hammer area is {}", hammer_.size.area());

  if (armors.size() > 0 && hammer_.size.area() > 0) {
    buff_.SetTarget(armors[0]);
    for (auto armor : armors) {
      if (cv::norm(hammer_.center - armor.SurfaceCenter()) <
          cv::norm(hammer_.center - buff_.GetTarget().SurfaceCenter())) {
        buff_.SetTarget(armor);
        // TODO
        if (circumference_.size() <= 5)
          circumference_.emplace_back(armor.SurfaceCenter());
      }
    }
    buff_.SetArmors(armors);
  } else {
    SPDLOG_WARN("can't find buff_armor");
  }
  const auto stop = high_resolution_clock::now();
  duration_armors_ = duration_cast<std::chrono::milliseconds>(stop - start);
}

void BuffDetector::MatchPredict() {
  buff_.SetPridict(Armor());
  if (cv::Point2f(0, 0) == buff_.GetCenter()) return;
  if (cv::Point2f(0, 0) == buff_.GetTarget().SurfaceCenter()) return;

  cv::Point2f target_center = buff_.GetTarget().SurfaceCenter();
  cv::Point2f center = buff_.GetCenter();
  SPDLOG_WARN("center is {},{}", buff_.GetCenter().x, buff_.GetCenter().y);
  common::Direction direction = buff_.GetDirection();
  Armor predict;

  double angle = Angle(target_center, center);
  double theta = DeltaTheta(buff_.GetTime());
  while (angle > 90) angle -= 90;
  if (direction == common::Direction::kCW) theta = -theta;
  double predict_angle = angle + theta;

  theta = theta / 180 * CV_PI;
  cv::Matx22d rot(cos(theta), -sin(theta), sin(theta), cos(theta));
  cv::Matx21d vec(target_center.x - center.x, target_center.y - center.y);
  cv::Matx21d point = rot * vec;
  cv::Point2f predict_center(point.val[0] + center.x, point.val[1] + center.y);
  cv::Size2d predict_size = rects_.back().size;

  SPDLOG_WARN("predict_center is {}, {}", predict_center.x, predict_center.y);
  cv::RotatedRect predict_rect(predict_center, predict_size, predict_angle);
  buff_.SetPridict(Armor(predict_rect));
}

void BuffDetector::VisualizeArmors(const cv::Mat &output, bool add_lable) {
  std::vector<Armor> armors = buff_.GetArmors();
  if (!armors.empty()) {
    for (auto &armor : armors) {
      auto vertices = armor.SurfaceVertices();
      if (vertices == buff_.GetTarget().SurfaceVertices()) continue;
      for (std::size_t i = 0; i < vertices.size(); ++i)
        cv::line(output, vertices[i], vertices[(i + 1) % 4], kGREEN);

      cv::drawMarker(output, armor.SurfaceCenter(), kGREEN, cv::MARKER_DIAMOND);

      if (add_lable) {
        std::ostringstream buf;
        buf << armor.SurfaceCenter().x << ", " << armor.SurfaceCenter().y;
        cv::putText(output, buf.str(), vertices[1], kCV_FONT, 1.0, kGREEN);
      }
    }
  }

  Armor target = buff_.GetTarget();
  if (cv::Point2f(0, 0) != target.SurfaceCenter()) {
    auto vertices = target.SurfaceVertices();
    for (std::size_t i = 0; i < vertices.size(); ++i)
      cv::line(output, vertices[i], vertices[(i + 1) % 4], kRED);
    cv::drawMarker(output, target.SurfaceCenter(), kRED, cv::MARKER_DIAMOND);
    if (add_lable) {
      std::ostringstream buf;
      buf << target.SurfaceCenter().x << ", " << target.SurfaceCenter().y;
      cv::putText(output, buf.str(), vertices[1], kCV_FONT, 1.0, kRED);
    }
  }
  Armor predict = buff_.GetPredict();
  auto vertices = predict.SurfaceVertices();
  for (std::size_t i = 0; i < vertices.size(); ++i)
    cv::line(output, vertices[i], vertices[(i + 1) % 4], kYELLOW, 8);
}

BuffDetector::BuffDetector() { SPDLOG_TRACE("Constructed."); }

BuffDetector::BuffDetector(const std::string &params_path,
                           game::Team buff_team) {
  LoadParams(params_path);
  buff_.SetTeam(buff_team);
  SPDLOG_TRACE("Constructed.");
}

BuffDetector::~BuffDetector() { SPDLOG_TRACE("Destructed."); }

const std::vector<Armor> &BuffDetector::Detect(
    const cv::Mat &frame) {
  SPDLOG_DEBUG("Detecting");
  FindRects(frame);
  MatchArmors();
  // MatchDirection();
  // MatchPredict();
  SPDLOG_DEBUG("Detected.");
  targets_.clear();
  targets_.emplace_back(buff_.GetTarget());
  return targets_;
}

void BuffDetector::VisualizeResult(const cv::Mat &output, int verbose) {
  SPDLOG_DEBUG("Visualizeing Result.");
  if (verbose > 10) {
    cv::drawContours(output, contours_, -1, kRED);
    cv::drawContours(output, contours_poly_, -1, kYELLOW);
  }

  if (verbose > 1) {
    int baseLine, v_pos = 0;

    std::string label =
        cv::format("%ld armors in %ld ms.", buff_.GetArmors().size(),
                   duration_armors_.count());
    cv::Size text_size = cv::getTextSize(label, kCV_FONT, 1.0, 2, &baseLine);
    v_pos += static_cast<int>(1.3 * text_size.height);
    cv::putText(output, label, cv::Point(0, v_pos), kCV_FONT, 1.0, kGREEN);

    label = cv::format("%ld rects in %ld ms.", rects_.size(),
                       duration_rects_.count());
    text_size = cv::getTextSize(label, kCV_FONT, 1.0, 2, &baseLine);
    v_pos += static_cast<int>(1.3 * text_size.height);
    cv::putText(output, label, cv::Point(0, v_pos), kCV_FONT, 1.0, kGREEN);
  }
  if (verbose > 3) {
    cv::Point2f vertices[4];
    hammer_.points(vertices);
    for (std::size_t i = 0; i < 4; ++i)
      cv::line(output, vertices[i], vertices[(i + 1) % 4], kRED);

    cv::drawMarker(output, buff_.GetCenter(), kRED, cv::MARKER_DIAMOND);
    cv::drawMarker(output, buff_.GetCenter(), kRED, cv::MARKER_DIAMOND);
  }
  VisualizeArmors(output, verbose);
  SPDLOG_DEBUG("Visualized.");
}