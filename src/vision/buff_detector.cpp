#include "buff_detector.hpp"

#include <ostream>

#include "compensator.hpp"
#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

namespace {

const auto kCV_FONT = cv::FONT_HERSHEY_SIMPLEX;
const cv::Scalar kGREEN(0., 255., 0.);
const cv::Scalar kRED(0., 0., 255.);
const cv::Scalar kYELLOW(0., 255., 255.);

const int kR = 1400;

}  // namespace

void BuffDetector::InitDefaultParams(const std::string &params_path) {
  cv::FileStorage fs(params_path,
                     cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);

  fs << "binary_th" << 220;
  fs << "se_erosion" << 2;
  fs << "ap_erosion" << 1.;
  fs << "se_anchor" << 2;

  fs << "contour_size_low_th" << 2;
  fs << "contour_area_low_th" << 100;
  fs << "contour_area_high_th" << 6000;
  fs << "rect_area_low_th" << 900;
  fs << "rect_area_high_th" << 9000;
  fs << "rect_ratio_low_th" << 0.4;
  fs << "rect_ratio_high_th" << 2.5;

  fs << "contour_center_area_low_th" << 100;
  fs << "contour_center_area_high_th" << 1000;
  fs << "rect_center_ratio_low_th" << 0.8;
  fs << "rect_center_ratio_high_th" << 1.25;
  fs << "rect_armor_area_low_th" << 900;
  fs << "rect_armor_area_high_th" << 3000;

  SPDLOG_DEBUG("Inited params.");
}

bool BuffDetector::PrepareParams(const std::string &params_path) {
  cv::FileStorage fs(params_path,
                     cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
  if (fs.isOpened()) {
    params_.binary_th = fs["binary_th"];
    params_.se_erosion = fs["se_erosion"];
    params_.ap_erosion = fs["ap_erosion"];
    params_.se_anchor = fs["se_anchor"];

    params_.contour_size_low_th = int(fs["contour_size_low_th"]);
    params_.contour_area_low_th = fs["contour_area_low_th"];
    params_.contour_area_high_th = fs["contour_area_high_th"];
    params_.rect_area_low_th = fs["rect_area_low_th"];
    params_.rect_area_high_th = fs["rect_area_high_th"];
    params_.rect_ratio_low_th = fs["rect_ratio_low_th"];
    params_.rect_ratio_high_th = fs["rect_ratio_high_th"];

    params_.contour_center_area_low_th = fs["contour_center_area_low_th"];
    params_.contour_center_area_high_th = fs["contour_center_area_high_th"];
    params_.rect_center_ratio_low_th = fs["rect_center_ratio_low_th"];
    params_.rect_center_ratio_high_th = fs["rect_center_ratio_high_th"];
    params_.rect_armor_area_low_th = fs["rect_armor_area_low_th"];
    params_.rect_armor_area_high_th = fs["rect_armor_area_high_th"];
    return true;
  } else {
    SPDLOG_ERROR("Can not load params.");
    return false;
  }
}

BuffDetector::BuffDetector() { SPDLOG_TRACE("Constructed."); }

BuffDetector::BuffDetector(const std::string &params_path,
                           game::Team buff_team) {
  LoadParams(params_path);
  buff_.SetTeam(buff_team);
  SPDLOG_TRACE("Constructed.");
}

BuffDetector::~BuffDetector() { SPDLOG_TRACE("Destructed."); }

void BuffDetector::FindRects(const cv::Mat &frame) {
  const auto start = std::chrono::high_resolution_clock::now();
  rects_.clear();

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
  img = channels[0];
#endif

  cv::threshold(img, img, 0, 255, cv::THRESH_OTSU);
  cv::Mat kernel = cv::getStructuringElement(
      cv::MORPH_RECT,
      cv::Size2i(2 * params_.se_erosion + 1, 2 * params_.se_erosion + 1),
      cv::Point(params_.se_erosion, params_.se_erosion));

  cv::dilate(img, img, kernel);
  cv::morphologyEx(img, img, cv::MORPH_CLOSE, kernel);
  cv::findContours(img, contours_, cv::MORPH_CLOSE, cv::CHAIN_APPROX_NONE);
  contours_poly_.resize(contours_.size());
  for (size_t k = 0; k < contours_.size(); ++k) {
    cv::approxPolyDP(cv::Mat(contours_[k]), contours_poly_[k],
                     params_.ap_erosion, true);
  }

  for (const auto &contour : contours_poly_) {
    if (contour.size() < params_.contour_size_low_th) continue;

    cv::RotatedRect rect = cv::minAreaRect(contour);
    double rect_ratio = rect.size.aspectRatio();
    double contour_area = cv::contourArea(contour);
    if (rect_ratio < params_.rect_ratio_low_th) continue;
    if (rect_ratio > params_.rect_ratio_high_th) continue;

    if (contour_area > params_.contour_area_high_th) continue;

    double rect_area = rect.size.area();
    if (rect_area < params_.rect_area_low_th) continue;
    if (rect_area > params_.rect_area_high_th) continue;

    rects_.emplace_back(rect);
  }

  const auto stop = std::chrono::high_resolution_clock::now();
  duration_rects_ =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
}

void BuffDetector::FindCenter() {
  const auto start = std::chrono::high_resolution_clock::now();
  buff_.SetCenter(cv::RotatedRect());

  for (const auto &contour : contours_poly_) {
    cv::RotatedRect rect = cv::minAreaRect(contour);
    double rect_ratio = rect.size.aspectRatio();
    double contour_area = cv::contourArea(contour);
    if (params_.contour_center_area_low_th < contour_area &&
        contour_area < params_.contour_center_area_high_th) {
      if (params_.rect_center_ratio_low_th < rect_ratio &&
          rect_ratio < params_.rect_center_ratio_high_th) {
        buff_.SetCenter(rect);
        continue;
      }
    }
  }

  const auto stop = std::chrono::high_resolution_clock::now();
  duration_center_ =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
}

void BuffDetector::FindTrack(const std::string &path) {
  const auto start = std::chrono::high_resolution_clock::now();
  buff_.SetTracks(std::vector<cv::RotatedRect>());

  double speed = buff_.GetSpeed();
  Armor armor = buff_.GetTarget();
  Compensator compensator(path);
  cv::Point3f center_coord = compensator.EstimateWorldCoord(armor);
  if (buff_.GetTracks().size() > 2) {
    // if(armor.SurfaceCenter().y<buff_.GetCenter().center.y){if()}
  }
  // TODO

  SPDLOG_DEBUG("Found tracks: {}", buff_.GetTracks().size());

  const auto stop = std::chrono::high_resolution_clock::now();
  duration_tracks_ =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  SPDLOG_DEBUG("duration_track_: {} ms", duration_tracks_.count());
}

double Dist(cv::Point2f a, cv::Point2f b) {
  return sqrt(powf(a.x - b.x, 2) + powf(a.y - b.y, 2));
}

void BuffDetector::MatchClockWise() {}

void BuffDetector::MatchSpeed() {}

void BuffDetector::MatchArmors() {
  const auto start = std::chrono::high_resolution_clock::now();
  buff_.SetArmors(std::vector<Armor>());

  cv::RotatedRect hammer;
  std::vector<Armor> armor_vec;

  for (auto &rect : rects_) {
    double rect_area = rect.size.area();
    SPDLOG_DEBUG("find area is {}", rect_area);
    if (rect_area > 1.5 * params_.rect_armor_area_high_th) {
      hammer = rect;
      continue;
    }
    if (rect_area < params_.rect_armor_area_low_th) continue;
    if (rect_area > params_.rect_armor_area_high_th) continue;
    armor_vec.emplace_back(Armor(rect));
  }
  SPDLOG_DEBUG("armor_vec.size is {}", armor_vec.size());
  SPDLOG_DEBUG("the buff's hammer area is {}", hammer.size.area());
  if (armor_vec.size() > 0 && hammer.size.area()) {
    buff_.SetTarget(armor_vec[0]);
    for (auto armor : armor_vec) {
      if (Dist(hammer.center, armor.SurfaceCenter()) <
          Dist(buff_.GetTarget().SurfaceCenter(), hammer.center)) {
        buff_.SetTarget(armor);
      }
    }
    buff_.SetArmors(armor_vec);
  } else {
    SPDLOG_WARN("can't find buff_armor");
  }
  const auto stop = std::chrono::high_resolution_clock::now();
  duration_armors_ =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
}

void BuffDetector::VisualizeArmor(const cv::Mat &output, bool add_lable) {
  std::vector<Armor> armors = buff_.GetArmors();
  if (!armors.empty()) {
    for (auto &armor : armors) {
      auto vertices = armor.SurfaceVertices();
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
  auto vertices = target.SurfaceVertices();
  for (std::size_t i = 0; i < vertices.size(); ++i)
    cv::line(output, vertices[i], vertices[(i + 1) % 4], kRED);
}

void BuffDetector::VisualizeTrack(const cv::Mat &output, bool add_lable) {
  std::vector<cv::RotatedRect> rects = buff_.GetTracks();
  if (!rects.empty()) {
    for (const auto &rect : rects) {
      std::vector<cv::Point2f> vertices(4);
      rect.points(vertices.data());

      for (std::size_t i = 0; i < vertices.size(); ++i)
        cv::line(output, vertices[i], vertices[(i + 1) % 4], kGREEN);

      cv::drawMarker(output, rect.center, kGREEN, cv::MARKER_DIAMOND);

      if (add_lable) {
        std::ostringstream buf;
        buf << rect.center.x << ", " << rect.center.y;
        cv::putText(output, buf.str(), vertices[1], kCV_FONT, 1.0, kGREEN);
      }
    }
  }
}

const std::vector<Buff> &BuffDetector::Detect(const cv::Mat &frame) {
  SPDLOG_DEBUG("Detecting");
  FindRects(frame);
  MatchArmors();
  SPDLOG_DEBUG("Detected.");
  targets_.clear();
  targets_.emplace_back(buff_);
  return targets_;
}

void BuffDetector::VisualizeResult(const cv::Mat &output, int verbose) {
  SPDLOG_DEBUG("Visualizeing Result.");
  if (verbose > 0) {
    cv::drawContours(output, contours_, -1, kRED);
    cv::drawContours(output, contours_poly_, -1, kYELLOW);
  }

  if (verbose > 1) {
    int baseLine, v_pos = 0;

    std::string label = cv::format("%ld armors in %ld ms.", targets_.size(),
                                   duration_armors_.count());
    cv::Size text_size = cv::getTextSize(label, kCV_FONT, 1.0, 2, &baseLine);
    v_pos += static_cast<int>(1.3 * text_size.height);
    cv::putText(output, label, cv::Point(0, v_pos), kCV_FONT, 1.0, kGREEN);

    label = cv::format("%ld rects in %ld ms.", rects_.size(),
                       duration_rects_.count());
    text_size = cv::getTextSize(label, kCV_FONT, 1.0, 2, &baseLine);
    v_pos += static_cast<int>(1.3 * text_size.height);
    cv::putText(output, label, cv::Point(0, v_pos), kCV_FONT, 1.0, kGREEN);

    label = cv::format("%ld tracks in %ld ms.", buff_.GetTracks().size(),
                       duration_tracks_.count());
    text_size = cv::getTextSize(label, kCV_FONT, 1.0, 2, &baseLine);
    v_pos += static_cast<int>(1.3 * text_size.height);
    cv::putText(output, label, cv::Point(0, v_pos), kCV_FONT, 1.0, kGREEN);
  }
  VisualizeArmor(output, verbose);
  VisualizeTrack(output, verbose);
  SPDLOG_DEBUG("Visualized.");
}