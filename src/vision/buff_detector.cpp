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

  fs << "contour_size_low_th" << 20;
  fs << "contour_area_low_th" << 100;
  fs << "contour_area_high_th" << 10000;
  fs << "bar_area_low_th" << 100;
  fs << "bar_area_high_th" << 10000;
  fs << "angle_high_th" << 60;
  fs << "aspect_ratio_low_th" << 2;
  fs << "aspect_ratio_high_th" << 6;

  fs << "angle_diff_th" << 10;
  fs << "length_diff_th" << 0.2;
  fs << "height_diff_th" << 0.2;
  fs << "area_diff_th" << 0.6;
  fs << "center_dist_low_th" << 1;
  fs << "center_dist_high_th" << 4;
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
    params_.bar_area_low_th = fs["bar_area_low_th"];
    params_.bar_area_high_th = fs["bar_area_high_th"];
    params_.angle_high_th = fs["angle_high_th"];
    params_.aspect_ratio_low_th = fs["aspect_ratio_low_th"];
    params_.aspect_ratio_high_th = fs["aspect_ratio_high_th"];

    params_.angle_diff_th = fs["angle_diff_th"];
    params_.length_diff_th = fs["length_diff_th"];
    params_.height_diff_th = fs["height_diff_th"];
    params_.area_diff_th = fs["area_diff_th"];
    params_.center_dist_low_th = fs["center_dist_low_th"];
    params_.center_dist_high_th = fs["center_dist_high_th"];
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

void BuffDetector::FindArmors(const cv::Mat &frame) {
  const auto start = std::chrono::high_resolution_clock::now();
  armors_.clear();

  frame_size_ = cv::Size(frame.cols, frame.rows);

  cv::Mat channels[3];
  cv::split(frame, channels);
  cv::Mat img = channels[0] - channels[2];
  // TODO

  const auto stop = std::chrono::high_resolution_clock::now();
  duration_armors_ =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
}

void BuffDetector::FindContours(const cv::Mat &frame) {
  const auto start = std::chrono::high_resolution_clock::now();
  buff_.SetContours(std::vector<std::vector<cv::Point2f>>());

  frame_size_ = cv::Size(frame.cols, frame.rows);

  std::vector<std::vector<cv::Point2f>> contours;

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
  cv::findContours(img, contours, cv::MORPH_CLOSE, cv::CHAIN_APPROX_NONE);

  buff_.SetContours(contours);

  const auto stop = std::chrono::high_resolution_clock::now();
  duration_contours_ =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
}

void BuffDetector::FindTrack(const cv::Mat &frame) {
  const auto start = std::chrono::high_resolution_clock::now();
  buff_.SetRects(std::vector<cv::RotatedRect>());

  // TODO

  SPDLOG_DEBUG("Found tracks: {}", buff_.GetRects().size());

  const auto stop = std::chrono::high_resolution_clock::now();
  duration_track_ =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  SPDLOG_DEBUG("duration_track_: {} ms", duration_track_.count());
}

void BuffDetector::MatchArmors(const cv::Mat &frame) {
  // TODO
  SPDLOG_DEBUG("Buff has been fould.");
}

void BuffDetector::VisualizeArmor(const cv::Mat &output, bool add_lable) {
  if (!armors_.empty()) {
    for (auto &armor : armors_) {
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

void BuffDetector::VisualizeContour(const cv::Mat &output, bool add_lable) {
  std::vector<std::vector<cv::Point2f>> contours = buff_.GetContours();
  if (!contours.empty()) {
    for (auto &contour : contours) {
      if (add_lable) {
        // TODO
        cv::drawContours(output, contour, -1, kGREEN);
      }
    }
  }
}

void BuffDetector::VisualizeTrack(const cv::Mat &output, bool add_lable) {
  std::vector<cv::RotatedRect> rects = buff_.GetRects();
  if (!rects.empty()) {
    for (auto &rect : rects) {
      auto vertices = buff_.Vertices2D(rect);
      for (std::size_t i = 0; i < vertices.size(); ++i)
        cv::line(output, vertices[i], vertices[(i + 1) % 4], kGREEN);

      cv::drawMarker(output, buff_.Center2D(rect), kGREEN, cv::MARKER_DIAMOND);

      if (add_lable) {
        std::ostringstream buf;
        buf << buff_.Center2D(rect).x << ", " << buff_.Center2D(rect).y;
        cv::putText(output, buf.str(), vertices[1], kCV_FONT, 1.0, kGREEN);
      }
    }
  }
}

const std::vector<Buff> &BuffDetector::Detect(const cv::Mat &frame) {
  SPDLOG_DEBUG("Detecting");
  FindArmors(frame);
  FindContours(frame);
  SPDLOG_DEBUG("Detected.");
  return targets_;
}

void BuffDetector::VisualizeResult(const cv::Mat &output, int verbose) {
  SPDLOG_DEBUG("Visualizeing Result.");
  if (verbose > 0) {
    // TODO
  }

  if (verbose > 1) {
    std::ostringstream buf;
    int baseLine;
    int v_pos = 0;

    buf << armors_.size() << " armors in " << duration_armors_.count()
        << " ms.";
    cv::Size text_size =
        cv::getTextSize(buf.str(), kCV_FONT, 1.0, 2, &baseLine);
    v_pos += static_cast<int>(1.3 * text_size.height);
    cv::putText(output, buf.str(), cv::Point(0, v_pos), kCV_FONT, 1.0, kGREEN);

    buf.str(std::string());
    buf << buff_.GetContours().size() << " contours in "
        << duration_contours_.count() << " ms.";
    text_size = cv::getTextSize(buf.str(), kCV_FONT, 1.0, 2, &baseLine);
    v_pos += static_cast<int>(1.3 * text_size.height);
    cv::putText(output, buf.str(), cv::Point(0, v_pos), kCV_FONT, 1.0, kGREEN);

    buf.str(std::string());
    buf << buff_.GetRects().size() << " track in " << duration_track_.count()
        << " ms.";
    text_size = cv::getTextSize(buf.str(), kCV_FONT, 1.0, 2, &baseLine);
    v_pos += static_cast<int>(1.3 * text_size.height);
    cv::putText(output, buf.str(), cv::Point(0, v_pos), kCV_FONT, 1.0, kGREEN);
  }

  VisualizeArmor(output, verbose);
  VisualizeContour(output, verbose);
  VisualizeTrack(output, verbose);
  SPDLOG_DEBUG("Visualized.");
}