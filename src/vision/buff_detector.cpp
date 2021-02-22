#include "buff_detector.hpp"

#include <ostream>

#include "opencv2/opencv.hpp"
#include "spdlog/spdlog.h"

namespace {

const auto kCV_FONT = cv::FONT_HERSHEY_SIMPLEX;
const auto kGREEN = cv::Scalar(0., 255., 0.);
const auto kRED = cv::Scalar(0., 0., 255.);
const auto kYELLOW = cv::Scalar(0., 255., 255.);

}  // namespace

void BuffDetector::InitDefaultParams(const std::string &params_path) {
  cv::FileStorage fs(params_path,
                     cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);

  fs.writeComment("binary threshold");
  fs << "binary_th" << 220;
  fs << "se_erosion" << 5;
  fs << "ap_erosion" << 1.;

  fs << "contour_size_low_th" << 5;
  fs << "contour_area_low_th" << 100;
  fs << "contour_area_high_th" << 10000;
  fs << "rect_area_low_th" << 100;
  fs << "rect_area_high_th" << 500;
  fs << "rect_ratio_low_th" << 0.4;
  fs << "rect_ratio_high_th" << 2.5;

  SPDLOG_DEBUG("Inited params.");
}

bool BuffDetector::PrepareParams(const std::string &params_path) {
  cv::FileStorage fs(params_path,
                     cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
  if (fs.isOpened()) {
    params_.binary_th = fs["binary_th"];
    params_.se_erosion = fs["se_erosion"];
    params_.ap_erosion = fs["ap_erosion"];

    params_.contour_size_low_th = int(fs["contour_size_low_th"]);
    params_.contour_area_low_th = fs["contour_area_low_th"];
    params_.contour_area_high_th = fs["contour_area_high_th"];
    params_.rect_area_low_th = fs["rect_area_low_th"];
    params_.rect_area_high_th = fs["rect_area_high_th"];
    params_.rect_ratio_low_th = fs["rect_ratio_low_th"];
    params_.rect_ratio_high_th = fs["rect_ratio_high_th"];
    return true;
  } else {
    SPDLOG_ERROR("Can not load params.");
    return false;
  }
}

BuffDetector::BuffDetector() { SPDLOG_TRACE("Constructed."); }

BuffDetector::BuffDetector(const std::string &params_path) {
  LoadParams(params_path);
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
      cv::Size2i(2 * params_.se_erosion + 1, 2 * params_.se_erosion + 1));

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

    double rect_area = rect.size.area();
    if (rect_area < params_.rect_area_low_th) continue;
    if (rect_area > params_.rect_area_high_th) continue;

    double rect_ratio = rect.size.aspectRatio();
    if (rect_ratio < params_.rect_ratio_low_th) continue;
    if (rect_ratio > params_.rect_ratio_high_th) continue;

    rects_.emplace_back(rect);
  }

  const auto stop = std::chrono::high_resolution_clock::now();
  duration_rects_ =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
}

void BuffDetector::FindTrack(const cv::Mat &frame) {
  const auto start = std::chrono::high_resolution_clock::now();
  buff_.SetTracks(std::vector<cv::RotatedRect>());

  // TODO

  SPDLOG_DEBUG("Found tracks: {}", buff_.GetTracks().size());

  const auto stop = std::chrono::high_resolution_clock::now();
  duration_tracks_ =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  SPDLOG_DEBUG("duration_track_: {} ms", duration_tracks_.count());
}

void BuffDetector::MatchArmors(const cv::Mat &frame) {
  const auto start = std::chrono::high_resolution_clock::now();
  buff_.SetArmors(std::vector<Armor>());

  frame_size_ = cv::Size(frame.cols, frame.rows);

  cv::Mat channels[3];
  cv::split(frame, channels);
  cv::Mat img = channels[0] - channels[2];
  // TODO

  const auto stop = std::chrono::high_resolution_clock::now();
  duration_armors_ =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
}

void BuffDetector::VisualizeArmor(const cv::Mat &output, bool add_lable) {
  std::vector<Armor> armors = buff_.GetArmors();
  if (!armors.empty()) {
    for (auto &armor : armors) {
      auto vertices = armor.Vertices2D();
      for (std::size_t i = 0; i < vertices.size(); ++i)
        cv::line(output, vertices[i], vertices[(i + 1) % 4], kGREEN);

      cv::drawMarker(output, armor.Center2D(), kGREEN, cv::MARKER_DIAMOND);

      if (add_lable) {
        std::ostringstream buf;
        buf << armor.Center2D().x << ", " << armor.Center2D().y;
        cv::putText(output, buf.str(), vertices[1], kCV_FONT, 1.0, kGREEN);
      }
    }
  }
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
  MatchArmors(frame);
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
    std::ostringstream buf;
    int baseLine;
    int v_pos = 0;

    buf << buff_.GetArmors().size() << " armors in " << duration_armors_.count()
        << " ms.";
    cv::Size text_size =
        cv::getTextSize(buf.str(), kCV_FONT, 1.0, 2, &baseLine);
    v_pos += static_cast<int>(1.3 * text_size.height);
    cv::putText(output, buf.str(), cv::Point(0, v_pos), kCV_FONT, 1.0, kGREEN);

    buf.str(std::string());
    buf << rects_.size() << " rects in " << duration_rects_.count() << " ms.";
    text_size = cv::getTextSize(buf.str(), kCV_FONT, 1.0, 2, &baseLine);
    v_pos += static_cast<int>(1.3 * text_size.height);
    cv::putText(output, buf.str(), cv::Point(0, v_pos), kCV_FONT, 1.0, kGREEN);

    buf.str(std::string());
    buf << buff_.GetTracks().size() << " tracks in " << duration_tracks_.count()
        << " ms.";
    text_size = cv::getTextSize(buf.str(), kCV_FONT, 1.0, 2, &baseLine);
    v_pos += static_cast<int>(1.3 * text_size.height);
    cv::putText(output, buf.str(), cv::Point(0, v_pos), kCV_FONT, 1.0, kGREEN);
  }
  VisualizeArmor(output, verbose);
  VisualizeTrack(output, verbose);
  SPDLOG_DEBUG("Visualized.");
}